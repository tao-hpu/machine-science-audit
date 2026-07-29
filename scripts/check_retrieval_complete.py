#!/usr/bin/env python3
"""检索完整性扫描(逐贡献口径)——检索收尾/断网续跑前的验收。

对齐 retrieve_s2_slow.py 的续跑逻辑:一条贡献算"完成"当且仅当 s2_complete=True。
报三类:缺整篇(近邻文件不存在)、半篇(部分贡献缺/未完成)、齐(全 s2_complete)。
只读,不改任何东西;有缺口时打印可直接复制的续跑命令。

用法:
  python3 scripts/check_retrieval_complete.py --extract-dir data/extractions_human --nbr-dir data/neighbors_human
  python3 scripts/check_retrieval_complete.py --extract-dir data/extractions      --nbr-dir data/neighbors_fars_cito
退出码:0=零缺口;1=有缺口(便于脚本/CI 判断)。
"""
import argparse, json, pathlib, sys


def paper_ids(extract_dir):
    ids = []
    for f in sorted(pathlib.Path(extract_dir).glob('*.json')):
        if f.name.startswith('_'):
            continue
        ids.append(f.stem)
    return ids


def expected_cids(extract_dir, pid):
    j = json.load(open(pathlib.Path(extract_dir) / f'{pid}.json'))
    return [c['id'] for c in j.get('contributions', [])]


def done_cids(nbr_dir, pid):
    p = pathlib.Path(nbr_dir) / f'{pid}.json'
    if not p.exists():
        return None  # 整篇缺
    d = json.load(open(p))
    return {c['id'] for c in d.get('contributions', []) if c.get('s2_complete')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--extract-dir', required=True, help='抽取目录(定义应有的论文/贡献全集)')
    ap.add_argument('--nbr-dir', required=True, help='近邻输出目录(被检验对象)')
    ap.add_argument('--show', type=int, default=20, help='最多列出多少条缺口明细')
    a = ap.parse_args()

    pids = paper_ids(a.extract_dir)
    missing_paper, partial, complete = [], [], []
    total_exp = total_done = 0

    for pid in pids:
        exp = expected_cids(a.extract_dir, pid)
        total_exp += len(exp)
        done = done_cids(a.nbr_dir, pid)
        if done is None:
            missing_paper.append((pid, exp))
            continue
        total_done += len(done)
        gap = [c for c in exp if c not in done]
        if gap:
            partial.append((pid, gap))
        else:
            complete.append(pid)

    n = len(pids)
    print(f'== 检索完整性:{a.nbr_dir}  (对照 {a.extract_dir})')
    print(f'   论文:{len(complete)}/{n} 齐 | {len(partial)} 半篇 | {len(missing_paper)} 缺整篇')
    print(f'   贡献:{total_done}/{total_exp} s2_complete ({100*total_done/max(1,total_exp):.1f}%)')

    if missing_paper:
        show = missing_paper[:a.show]
        print(f'   缺整篇({len(missing_paper)}):' + ', '.join(p for p, _ in show) +
              (' …' if len(missing_paper) > a.show else ''))
    if partial:
        show = partial[:a.show]
        print(f'   半篇({len(partial)}):' + ', '.join(f'{p}[缺 {",".join(g)}]' for p, g in show) +
              (' …' if len(partial) > a.show else ''))

    gaps = len(missing_paper) + len(partial)
    if gaps:
        print(f'\n   ⚠️ {gaps} 篇有缺口。续跑(幂等,只补未完成的贡献):')
        cutoff = 'human' if 'human' in a.extract_dir else ('a4s' if 'a4s' in a.extract_dir else 'fars')
        print(f'   python3 -u scripts/retrieve_s2_slow.py --extract-dir {a.extract_dir} \\')
        print(f'       --out-dir {a.nbr_dir} --facet-alloc --cutoff-source {cutoff} --channel cito --oa api')
        sys.exit(1)
    else:
        print('\n   ✅ 零缺口,全部 s2_complete。')
        sys.exit(0)


if __name__ == '__main__':
    main()
