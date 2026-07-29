#!/usr/bin/env python3
"""检索查询策略消融(gate 覆盖率口径)——测 query-by-document 能否拉起 cito 召回。

背景:cito hybrid 用现有 v5 关键词短语查询只覆盖 23.9% 人选近邻,且 66.8% 两通道都漏。
但抽样证明漏的 82% 就在 cito 库里 → 瓶颈是查询,不是语料。本脚本比较查询形态:
  KW20  — 现有 v5 关键词查询 ×N @limit20(复用 data/neighbors_cito_val,gate 基线)
  DOC   — query-by-document:每条贡献 purpose+mechanism+evaluation 拼成一条语义查询
  DOCsp — 拆成 purpose/mechanism/evaluation 三条分别查
都走 cito hybrid + published_before(cutoff)。指标:覆盖 data/neighbors_facet_val 人选近邻标题,
同时报池大小(=judge 成本代理),因为大池天然覆盖高,要看「每单位池的覆盖效率」和「补进 KW 漏的」。

用法:python3 scripts/gate_query_ablation.py [--limit 100]
"""
import argparse, json, os, pathlib, re, sys, urllib.parse, urllib.request

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass
K = os.environ['CITO_API_KEY']; B = os.environ['CITO_API_BASE'].rstrip('/')

QUERIES = json.load(open('data/neighbors_s2/_queries.json'))
FACET = pathlib.Path('data/neighbors_facet_val')


def norm(t):
    return re.sub(r'[^a-z0-9]+', ' ', (t or '').lower()).strip()


def cutoff(pid):
    return json.load(open(f'data/fars-a-reviews/data/{pid}/paperreview.json'))['submission_date'][:10]


def kw_queries(pid, cid):
    return (QUERIES.get(f'{pid}/{cid}/v5') or QUERIES.get(f'{pid}/{cid}/v4')
            or QUERIES.get(f'{pid}/{cid}') or [])


def contribs(pid):
    return {c['id']: c for c in json.load(open(f'data/extractions/{pid}.json'))['contributions']}


def cito(q, cut, mode='hybrid', lim=100):
    q = q[:1200]
    p = {'q': q, 'mode': mode, 'limit': lim, 'enrich': 'true', 'published_before': cut}
    r = urllib.request.Request(f'{B}/search?' + urllib.parse.urlencode(p),
                               headers={'Authorization': f'Bearer {K}'})
    with urllib.request.urlopen(r, timeout=30) as resp:
        return json.loads(resp.read()).get('hits', [])


def pool_from(queries, cut, lim):
    """一组查询 → 去重标题集(norm)。"""
    seen = set()
    for q in queries:
        if not q:
            continue
        for h in cito(q, cut, 'hybrid', lim):
            if h.get('title'):
                seen.add(norm(h['title']))
    return seen


def existing_pool(d, pid):
    f = pathlib.Path(d) / f'{pid}.json'
    if not f.exists():
        return {}
    o = json.load(open(f))
    return {c['id']: {norm(n['title']) for n in c['neighbors']} for c in o['contributions']}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=100)
    args = ap.parse_args()

    # 累加器:每策略 (命中, gold总数, 池大小和)
    strat = {k: [0, 0, 0] for k in ['KW20', 'DOC', 'DOCsp', 'KW20+DOCsp']}
    uniq_over_kw = 0   # DOCsp 补进 KW20 漏的 gold 数
    n_contribs = 0

    for vf in sorted(FACET.glob('FA*.json')):
        val = json.load(open(vf)); pid = val['paper_id']; cut = cutoff(pid)
        cmap = contribs(pid)
        kw_existing = existing_pool('data/neighbors_cito_val', pid)  # KW20 gate 基线池
        for c in val['contributions']:
            cid = c['id']
            gold = {norm(n['title']) for n in c.get('neighbors', [])}
            if not gold:
                continue
            ct = cmap.get(cid, {})
            doc_txt = ' '.join(filter(None, [ct.get('purpose'), ct.get('mechanism'), ct.get('evaluation')]))
            fields = [ct.get('purpose'), ct.get('mechanism'), ct.get('evaluation')]

            pools = {
                'KW20': kw_existing.get(cid, set()),
                'DOC': pool_from([doc_txt], cut, args.limit) if doc_txt else set(),
                'DOCsp': pool_from(fields, cut, args.limit),
            }
            pools['KW20+DOCsp'] = pools['KW20'] | pools['DOCsp']

            for k, p in pools.items():
                strat[k][0] += len(gold & p)
                strat[k][1] += len(gold)
                strat[k][2] += len(p)
            uniq_over_kw += len((gold & pools['DOCsp']) - pools['KW20'])
            n_contribs += 1
        print(f'  {pid} 完成', flush=True)

    print(f'\n{"策略":<14}{"覆盖":>18}{"平均池/贡献":>12}')
    for k, (hit, tot, psz) in strat.items():
        print(f'{k:<14}{f"{hit}/{tot}={hit/tot:.1%}":>18}{psz/max(n_contribs,1):>12.0f}')
    print(f'\nDOCsp 补进 KW20 漏掉的 gold: {uniq_over_kw} 篇  (贡献数 {n_contribs})')
    out = {k: {'coverage': f'{v[0]}/{v[1]}={v[0]/v[1]:.1%}', 'pool_sum': v[2]} for k, v in strat.items()}
    out['docsp_unique_over_kw'] = uniq_over_kw
    out['limit'] = args.limit
    pathlib.Path('data/neighbors_cito_val/_query_ablation.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print('QUERY-ABLATION-DONE', flush=True)
