#!/usr/bin/env python3
"""坏近邻局部重跑(用 v5 新查询 prompt)。

依据 data/neighbor_relevance.json 的 ontopic_count 找出坏近邻贡献,把它们在
neighbors_s2/<pid>.json 里标记 s2_complete=false,再调用 retrieve_s2_slow.process
重做这些贡献 —— 因脚本已升 v5(带跨域护栏),查询会用新 prompt 重新生成。

⚠️ 必须在主检索完全跑完后运行(S2 key 1 req/s 全局共享,不能两个检索并发)。

用法:
  python3 scripts/rerun_bad_neighbors.py --threshold 1 --mark-only   # 只标记,先看清单
  python3 scripts/rerun_bad_neighbors.py --threshold 1               # 标记+重跑
"""
import json, os, sys, pathlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import scripts.retrieve_s2_slow as R

REL = pathlib.Path('data/neighbor_relevance.json')
NBR = pathlib.Path('data/neighbors_s2')


def bad_by_pid(threshold):
    rel = json.load(open(REL))
    byp = {}
    for v in rel.values():
        oc = v.get('ontopic_count', 9)
        if oc <= threshold:  # 含 -1(扫描报错)一并重跑,无害
            byp.setdefault(v['pid'], set()).add(v['cid'])
    return byp


def mark_incomplete(pid, cids):
    f = NBR / f'{pid}.json'
    d = json.load(open(f))
    n = 0
    for c in d['contributions']:
        if c['id'] in cids and c.get('s2_complete'):
            c['s2_complete'] = False
            n += 1
    json.dump(d, open(f, 'w'), ensure_ascii=False, indent=1)
    return n


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=int, default=1, help='ontopic_count <= 此值判坏')
    ap.add_argument('--mark-only', action='store_true')
    ap.add_argument('--k', type=int, default=20)
    a = ap.parse_args()
    byp = bad_by_pid(a.threshold)
    total = sum(len(v) for v in byp.values())
    print(f'坏近邻贡献: {total} 条,分布于 {len(byp)} 篇 (threshold ontopic<={a.threshold})')
    marked = 0
    for pid, cids in sorted(byp.items()):
        marked += mark_incomplete(pid, cids)
    print(f'标记 s2_complete=false: {marked} 条')
    if a.mark_only:
        print('--mark-only:已标记,未重跑。'); return
    print('开始重跑(v5 查询 prompt)...')
    for i, pid in enumerate(sorted(byp), 1):
        done = R.process(pid, a.k)
        print(f'[{i}/{len(byp)}] {pid} redone {done} contribs', flush=True)
    print('RERUN-DONE', flush=True)


if __name__ == '__main__':
    main()
