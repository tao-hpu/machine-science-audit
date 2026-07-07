#!/usr/bin/env python3
"""验证 per-facet 名额分配能否压下假 novel。

对 val2 的同一批贡献(FA0001-FA0006),用 FACET_ALLOC 重选近邻(候选池重新抓,
查询已缓存 v5),再过两裁判,和 val2(全局 top-k 选择)三态分布直接对比。
产物:data/neighbors_facet_val/、data/judgments_facet_val/。
"""
import json, os, sys, pathlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import scripts.retrieve_s2_slow as R
import scripts.judge_v3 as J
R.FACET_ALLOC = True  # 强制开 per-facet 分配

NBRD = pathlib.Path('data/neighbors_facet_val'); NBRD.mkdir(exist_ok=True)
JUDGE_MODELS = ['gpt-4o', 'claude-sonnet-4-6']
PIDS = ['FA0001', 'FA0002', 'FA0003', 'FA0004', 'FA0005', 'FA0006']


def consensus(states):
    from collections import Counter
    v = [s for s in states if s]
    if not v: return None
    top = Counter(v).most_common()
    return 'split' if len(top) > 1 and top[0][1] == top[1][1] else top[0][0]


def main():
    qc = R.load_query_cache()
    from collections import Counter
    old_st, new_st = Counter(), Counter()
    # 载入 val2 旧结果
    old = {}
    for f in pathlib.Path('data/judgments_val2').glob('FA*.json'):
        for c in json.load(open(f))['contributions']:
            old[f'{f.stem}/{c["cid"]}'] = c['consensus_state']
    for pid in PIDS:
        ext = json.load(open(f'data/extractions/{pid}.json'))
        cutoff = R.cutoff_for(pid)
        out = {'paper_id': pid, 'contributions': []}
        for c in ext['contributions']:
            key = f'{pid}/{c["id"]}'
            if key not in old:
                continue
            r = R.retrieve_one(pid, c, cutoff, qc, 20)
            out['contributions'].append(r)
            nbrs = r['neighbors']
            states = []
            for m in JUDGE_MODELS:
                try:
                    states.append(J.judge_one(m, c, nbrs)['state'])
                except Exception as e:
                    states.append(None)
            cs = consensus(states)
            ov = old.get(key)
            old_st[ov] += 1; new_st[cs] += 1
            # facet 覆盖标签统计
            covered_facets = set()
            for h in nbrs:
                for ff in h.get('facet_for', []):
                    covered_facets.add(ff)
            flag = '  <<<' if ov == 'facet-novel' and cs != 'facet-novel' else ''
            print(f'{key:12} old={str(ov):13} new={str(cs):13} facets_in_nbrs={sorted(covered_facets)}{flag}', flush=True)
        json.dump(out, open(NBRD / f'{pid}.json', 'w'), ensure_ascii=False, indent=1)
    print(f'\n旧(全局topk): {dict(old_st)}')
    print(f'新(perfacet): {dict(new_st)}')


if __name__ == '__main__':
    main()
