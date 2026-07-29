#!/usr/bin/env python3
"""V1 召回打分:gold 先行工作是否出现在判定时使用的近邻列表里。

用法: python3 scripts/v1_gold_recall.py [--pairs data/v1_gold/gold_pairs.json] \
        [--out data/v1_gold/recall_report.json]

口径:
  - 近邻列表 = rescue 目录里该贡献的 neighbors(判定层实际看到的集合,文件序);
  - 命中 = gold 标题与某近邻标题归一化相似度 >= 0.85;
  - pair-level recall = 命中的 gold 对 / 全部可用 gold 对;
  - contribution-level recall = 至少命中一条 gold 的贡献 / 有 gold 的贡献;
  - recall@10 / @20 / @full 按近邻文件序截断;双臂分开报。
"""
import argparse
import difflib
import json
import re

NBR = {'machine': 'data/neighbors_fars_cito_rescue', 'human': 'data/neighbors_human_rescue'}
KS = (10, 20, None)  # None = full list


def norm(t):
    t = re.sub(r'[^a-z0-9 ]', ' ', (t or '').lower())
    return re.sub(r'\s+', ' ', t).strip()


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def neighbors_for(arm, pid, cid):
    d = json.load(open(f'{NBR[arm]}/{pid}.json'))
    for c in d['contributions']:
        if c['id'] == cid:
            return c['neighbors']
    return []


def hit_rank(gold_title, nbrs, thresh=0.85):
    for i, n in enumerate(nbrs):
        if sim(gold_title, n['title']) >= thresh:
            return i + 1, n['title']
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default='data/v1_gold/gold_pairs.json')
    ap.add_argument('--out', default='data/v1_gold/recall_report.json')
    a = ap.parse_args()

    data = json.load(open(a.pairs))
    usable = [p for p in data['pairs']
              if p['oa_status'] == 'found' and p.get('before_cutoff')
              and not p.get('rejected')]  # Tao 抽查否决的对

    detail, per_arm = [], {}
    for p in usable:
        nbrs = neighbors_for(p['arm'], p['paper_id'], p['cid'])
        rank, matched = hit_rank(p['title'], nbrs)
        # paper 级:gold 是否落在同论文任一贡献的近邻里(诊断 facet 分配 vs 检索)
        paper_hit = rank is not None
        if not paper_hit:
            d = json.load(open(f'{NBR[p["arm"]]}/{p["paper_id"]}.json'))
            all_nbrs = [n for c in d['contributions'] for n in c['neighbors']]
            paper_hit = hit_rank(p['title'], all_nbrs)[0] is not None
        detail.append({
            'arm': p['arm'], 'paper_id': p['paper_id'], 'cid': p['cid'],
            'source': p.get('source'), 'gold_title': p['title'],
            'gold_date': p.get('oa_date'), 'facet': p.get('facet'),
            'n_neighbors': len(nbrs), 'hit_rank': rank,
            'matched_neighbor': matched, 'paper_level_hit': paper_hit,
        })

    report = {'n_usable_pairs': len(usable), 'arms': {}}
    for arm in ('machine', 'human'):
        rows = [d for d in detail if d['arm'] == arm]
        if not rows:
            continue
        arm_rep = {'n_pairs': len(rows)}
        for k in KS:
            key = f'recall@{k or "full"}'
            hits = [r for r in rows
                    if r['hit_rank'] is not None and (k is None or r['hit_rank'] <= k)]
            arm_rep[key] = round(len(hits) / len(rows), 3)
            arm_rep[key + '_n'] = len(hits)
        # contribution-level(至少一条 gold 命中)
        by_c = {}
        for r in rows:
            key = (r['paper_id'], r['cid'])
            by_c.setdefault(key, []).append(r['hit_rank'] is not None)
        arm_rep['n_contributions'] = len(by_c)
        arm_rep['contrib_recall@full'] = round(
            sum(any(v) for v in by_c.values()) / len(by_c), 3)
        arm_rep['paper_recall@full'] = round(
            sum(1 for r in rows if r['paper_level_hit']) / len(rows), 3)
        per_arm[arm] = arm_rep
    report['arms'] = per_arm
    report['detail'] = detail

    json.dump(report, open(a.out, 'w'), ensure_ascii=False, indent=2)
    for arm, r in per_arm.items():
        print(f'{arm}: pairs={r["n_pairs"]} recall@10={r["recall@10"]} '
              f'recall@20={r["recall@20"]} recall@full={r["recall@full"]} '
              f'contrib@full={r["contrib_recall@full"]} (n_contrib={r["n_contributions"]})')
    misses = [d for d in detail if d['hit_rank'] is None]
    print(f'misses: {len(misses)}/{len(detail)}')
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
