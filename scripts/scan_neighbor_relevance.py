#!/usr/bin/env python3
"""摸底:检索近邻在不在题上。

对每条已检索完成(s2_complete)的贡献,用 gpt-4o 一次判:给贡献的
purpose/mechanism/domain + top-6 近邻标题,问这些近邻是否本贡献的相关先行工作。
产出 data/neighbor_relevance.json:每条 {pid,cid,ontopic_count(0-6),verdict,note}。
verdict=bad(ontopic<=1)的进重跑清单。断点续跑:已判的跳过。

用法:
  python3 scripts/scan_neighbor_relevance.py           # 扫描(默认 FARS v5 目录)
  python3 scripts/scan_neighbor_relevance.py --status   # 看进度+坏比例
  # 换语料/换近邻版本(注意 --out 必须换新文件,否则 pid/cid 撞 key 会被跳过):
  python3 scripts/scan_neighbor_relevance.py --nbr data/neighbors_s2_v6 --out data/neighbor_relevance_v6.json
"""
import argparse, json, os, sys, time, pathlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import scripts.judge_v3 as J

EXT = pathlib.Path('data/extractions')
NBR = pathlib.Path('data/neighbors_s2')
OUT = pathlib.Path('data/neighbor_relevance.json')

PROMPT = """A research contribution and its retrieved "nearest prior work" are below. Judge how many of the retrieved papers are genuinely ON-TOPIC prior art for THIS contribution (same research problem or a true cross-domain analog of the same core idea — NOT merely sharing a generic tool/buzzword).

CONTRIBUTION:
- purpose: {purpose}
- mechanism: {mechanism}
- domain: {domain}

RETRIEVED PAPERS (titles):
{titles}

Output strict JSON only: {{"ontopic_count": <int 0-6>, "note": "<=12 words"}}"""


def load_out():
    return json.load(open(OUT)) if OUT.exists() else {}


def main():
    global EXT, NBR, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--ext', default=str(EXT)); ap.add_argument('--nbr', default=str(NBR))
    ap.add_argument('--out', default=str(OUT)); ap.add_argument('--status', action='store_true')
    a = ap.parse_args()
    EXT, NBR, OUT = pathlib.Path(a.ext), pathlib.Path(a.nbr), pathlib.Path(a.out)
    res = load_out()
    if a.status:
        vals = list(res.values())
        bad = [v for v in vals if v['ontopic_count'] <= 1]
        print(f'scanned: {len(vals)} | bad(ontopic<=1): {len(bad)} ({100*len(bad)/max(1,len(vals)):.0f}%)')
        from collections import Counter
        print('ontopic_count dist:', dict(sorted(Counter(v['ontopic_count'] for v in vals).items())))
        return
    n = 0
    for f in sorted(f for f in NBR.glob('*.json') if not f.name.startswith('_')):
        pid = f.stem
        ext = {c['id']: c for c in json.load(open(EXT / f'{pid}.json'))['contributions']}
        for c in json.load(open(f))['contributions']:
            if not c.get('s2_complete'):
                continue
            key = f'{pid}/{c["id"]}'
            if key in res:
                continue
            ec = ext.get(c['id'])
            if not ec:
                continue
            titles = '\n'.join(f"[{i+1}] {nb['title'][:90]}" for i, nb in enumerate(c['neighbors'][:6]))
            try:
                out = J.chat('gpt-4o', PROMPT.format(purpose=ec['purpose'][:200], mechanism=ec['mechanism'][:200],
                                                     domain=ec['domain'][:100], titles=titles), 120)
                j = J.parse_json(out) or {}
                res[key] = {'pid': pid, 'cid': c['id'], 'ontopic_count': int(j.get('ontopic_count', -1)),
                            'note': j.get('note', ''), 'verdict': 'bad' if int(j.get('ontopic_count', 9)) <= 1 else 'ok'}
            except Exception as e:
                res[key] = {'pid': pid, 'cid': c['id'], 'ontopic_count': -1, 'note': f'ERR {repr(e)[:40]}', 'verdict': 'err'}
            n += 1
            if n % 20 == 0:
                json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
                print(f'  ...{len(res)} scanned', flush=True)
            time.sleep(0.3)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(f'SCAN-DONE {len(res)}', flush=True)


if __name__ == '__main__':
    main()
