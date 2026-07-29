#!/usr/bin/env python3
"""V1 gold 验证:OpenAlex 核实提名的先行工作真实存在且早于 cutoff。

用法: python3 scripts/v1_verify_gold.py --noms <dir with batch_*.json> \
        [--out data/v1_gold/gold_pairs.json]

对每条提名:
  1. OpenAlex title.search 查该标题(polite pool);
  2. 归一化标题相似度匹配返回结果(防同名近似);
  3. 命中则记录 openalex id / DOI / publication_date,核对 date < 该论文 cutoff;
  4. 机器臂查不到的按幻觉引用剔除并单独记数(本身是可报的副产品)。

手工种子(data/gold_neighbors/)另行并入:每篇取前 3 条(第 4 条起是当年
判定层验证的干扰项,不算 gold),标 source=hand。
"""
import argparse
import difflib
import glob
import json
import os
import re
import time
import urllib.parse

import requests

OA = 'https://api.openalex.org/works'
S2 = 'https://api.semanticscholar.org/graph/v1/paper/search'
MAILTO = 'v1-audit@example.org'
SAMPLE = 'data/v1_gold/sample.json'
NBR = {'machine': 'data/neighbors_fars_cito_rescue', 'human': 'data/neighbors_human_rescue'}
SEED_GOLD_DIR = 'data/gold_neighbors'
SEED_KEEP = 3  # 每个手工文件前 N 条是真 gold


def norm(t):
    t = re.sub(r'[^a-z0-9 ]', ' ', (t or '').lower())
    return re.sub(r'\s+', ' ', t).strip()


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def _load_env():
    if not os.path.exists('.env'):
        return
    for line in open('.env'):
        line = line.strip()
        if line and not line.startswith('#') and '=' in line and "'" not in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k, v)


def oa_lookup(title, session):
    q = urllib.parse.quote(norm(title))
    url = f'{OA}?filter=title.search:{q}&per-page=5&mailto={MAILTO}'
    for attempt in range(3):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            body = r.json()
            if 'error' in body:  # OA 免费日预算耗尽时返回 200 + error body
                return None
            return body.get('results', [])
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
    return None  # 失败,区别于查无此文


def s2_lookup(title, session):
    """S2 兜底:OA 限流/未收录时用。返回与 OA results 同构的最小字段。"""
    key = os.environ.get('S2_API_KEY', '')
    for attempt in range(4):
        try:
            r = session.get(S2, params={'query': norm(title), 'limit': 5,
                                        'fields': 'title,publicationDate,year,externalIds'},
                            headers={'x-api-key': key} if key else {}, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            out = []
            for p in r.json().get('data', []):
                date = p.get('publicationDate') or (
                    f"{p['year']}-12-31" if p.get('year') else None)  # 只知年份取最保守日期
                out.append({'id': f"S2:{p.get('paperId', '')}",
                            'title': p.get('title'), 'publication_date': date})
            return out
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
    return None


def cutoff_of(arm, pid):
    d = json.load(open(f'{NBR[arm]}/{pid}.json'))
    return d.get('cutoff')


def verify_one(nom, arm, pid, session):
    title = nom.get('title') or ''
    res = oa_lookup(title, session)
    out = dict(nom)
    out.update({'oa_status': None, 'oa_id': None, 'oa_date': None,
                'oa_title': None, 'oa_sim': None, 'before_cutoff': None})
    best, best_s = None, 0.0
    for w in (res or []):
        s = sim(title, w.get('title') or w.get('display_name') or '')
        if s > best_s:
            best, best_s = w, s
    if best is None or best_s < 0.80:
        # OA 失败/限流/未收录 → S2 兜底(提名标题常带括号缩写,去掉再试)
        for t in (title, re.sub(r'\([^)]*\)', ' ', title)):
            res2 = s2_lookup(t, session)
            for w in (res2 or []):
                s = sim(t, w.get('title') or '')
                if s > best_s:
                    best, best_s = w, s
            if best_s >= 0.80:
                break
            time.sleep(1.1)  # S2 免费档限速
    if best is None or best_s < 0.80:
        out['oa_status'] = 'lookup_failed' if res is None else 'not_found'
        out['oa_sim'] = round(best_s, 3)
        return out
    out['oa_status'] = 'found'
    out['oa_id'] = best.get('id')
    out['oa_title'] = best.get('title') or best.get('display_name')
    out['oa_date'] = best.get('publication_date')
    out['oa_sim'] = round(best_s, 3)
    cut = cutoff_of(arm, pid)
    if out['oa_date'] and cut:
        out['before_cutoff'] = out['oa_date'] < cut
    return out


def recheck(out_path):
    """只复查上一轮 not_found / lookup_failed 的条目(OA 限流误伤),原地更新。"""
    session = requests.Session()
    data = json.load(open(out_path))
    n_fix = 0
    for i, p in enumerate(data['pairs']):
        if p['oa_status'] not in ('not_found', 'lookup_failed'):
            continue
        v = verify_one({k: p[k] for k in p if not k.startswith('oa_') and
                        k not in ('before_cutoff',)}, p['arm'], p['paper_id'], session)
        for k in ('oa_status', 'oa_id', 'oa_date', 'oa_title', 'oa_sim', 'before_cutoff'):
            p[k] = v[k]
        if p['oa_status'] == 'found':
            n_fix += 1
        print(f"  recheck {p['paper_id']}/{p['cid']} \"{p['title'][:50]}\" -> {p['oa_status']}")
        time.sleep(0.3)
    from collections import Counter
    data['stats'] = dict(Counter(p['oa_status'] for p in data['pairs']))
    data['stats']['after_cutoff'] = sum(1 for p in data['pairs']
                                        if p['oa_status'] == 'found' and p['before_cutoff'] is False)
    json.dump(data, open(out_path, 'w'), ensure_ascii=False, indent=2)
    ok = [p for p in data['pairs'] if p['oa_status'] == 'found' and p['before_cutoff']]
    print(f'recheck done, {n_fix} recovered; stats: {data["stats"]}')
    print(f'usable gold pairs (found & before cutoff): {len(ok)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--noms')
    ap.add_argument('--out', default='data/v1_gold/gold_pairs.json')
    ap.add_argument('--recheck', action='store_true',
                    help='只复查已有输出里 not_found/lookup_failed 的条目')
    a = ap.parse_args()
    _load_env()
    if a.recheck:
        recheck(a.out)
        return
    if not a.noms:
        ap.error('--noms is required unless --recheck')

    session = requests.Session()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.noms, 'batch_*.json'))):
        rows.extend(json.load(open(f)))
    print(f'{len(rows)} jobs loaded from {a.noms}')

    pairs, stats = [], {'found': 0, 'not_found': 0, 'lookup_failed': 0,
                        'after_cutoff': 0, 'empty_jobs': 0}
    for job in rows:
        arm, pid, cid = job['arm'], job['paper_id'], job['cid']
        noms = job.get('nominations', [])
        if not noms:
            stats['empty_jobs'] += 1
        for nom in noms:
            v = verify_one(nom, arm, pid, session)
            v.update({'arm': arm, 'paper_id': pid, 'cid': cid,
                      'source': 'citation'})
            stats[v['oa_status']] = stats.get(v['oa_status'], 0)
            stats[v['oa_status']] += 1
            if v['oa_status'] == 'found' and v['before_cutoff'] is False:
                stats['after_cutoff'] += 1
            pairs.append(v)
            time.sleep(0.15)
        print(f'  {pid}/{cid}: {len(noms)} noms verified')

    # 手工种子并入
    sample = json.load(open(SAMPLE))
    seeds = [r for r in sample['arms']['machine'] if r['source'] == 'hand-seed']
    for r in seeds:
        pid = r['paper_id']
        gfile = os.path.join(SEED_GOLD_DIR, f'{pid}.json')
        if not os.path.exists(gfile):
            continue
        for e in json.load(open(gfile))[:SEED_KEEP]:
            v = verify_one({'title': e['title'], 'year': (e.get('date') or '')[:4]},
                           'machine', pid, session)
            v.update({'arm': 'machine', 'paper_id': pid, 'cid': r['cid'],
                      'source': 'hand', 'facet': None,
                      'why': 'hand-confirmed gold (judge-layer validation set)'})
            pairs.append(v)
            time.sleep(0.15)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({'stats': stats, 'pairs': pairs},
              open(a.out, 'w'), ensure_ascii=False, indent=2)
    ok = [p for p in pairs if p['oa_status'] == 'found' and p['before_cutoff']]
    print(f'stats: {stats}')
    print(f'usable gold pairs (found & before cutoff): {len(ok)}')
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
