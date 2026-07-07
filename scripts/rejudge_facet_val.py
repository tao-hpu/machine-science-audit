#!/usr/bin/env python3
"""v6 验证批重判:v6 per-facet 近邻(data/neighbors_facet_val/,已缓存,不碰 S2)
× 修复后 judge_v3(确定性 finalize,治 FINALIZE 截断假 novel)× 两裁判。
产物 data/judgments_facet_val/<PID>.json,含 raw stage1/stage2 供 split 根因分析。
断点续跑:已判贡献跳过。
"""
import json, os, sys, time, pathlib
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import scripts.judge_v3 as J

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--out', default='data/judgments_facet_val')
_ap.add_argument('--models', default='gpt-4o,claude-sonnet-4-6')
_A = _ap.parse_args()
NBR = pathlib.Path('data/neighbors_facet_val')
OUT = pathlib.Path(_A.out); OUT.mkdir(exist_ok=True)
MODELS = _A.models.split(',')


def consensus(states):
    v = [s for s in states if s]
    if not v: return None
    top = Counter(v).most_common()
    return 'split' if len(top) > 1 and top[0][1] == top[1][1] else top[0][0]


def main():
    tally = Counter()
    for nf in sorted(NBR.glob('FA*.json')):
        pid = nf.stem
        nbrj = json.load(open(nf))
        ext = json.load(open(f'data/extractions/{pid}.json'))
        outf = OUT / f'{pid}.json'
        rows = json.load(open(outf))['contributions'] if outf.exists() else []
        done = {r['cid'] for r in rows}
        for nc in nbrj['contributions']:
            cid = nc['id']
            if cid in done: continue
            c = next(x for x in ext['contributions'] if x['id'] == cid)
            per_model = {}
            for m in MODELS:
                try:
                    r = J.judge_one(m, c, nc['neighbors'], keep_raw=True)
                    per_model[m] = {'state': r['state'], 'facets': r['facets'],
                                    'self_check': r['self_check_triggered'], 'raw': r['raw']}
                except Exception as ex:
                    per_model[m] = {'state': None, 'facets': None, 'error': repr(ex)[:120]}
                time.sleep(0.5)
            cs = consensus([per_model[m].get('state') for m in MODELS])
            rows.append({'cid': cid, 'type': c.get('type'), 'consensus_state': cs,
                         'prompt_version': J.PROMPT_VERSION, 'per_model': per_model})
            json.dump({'paper_id': pid, 'title': ext.get('title'), 'contributions': rows},
                      open(outf, 'w'), ensure_ascii=False, indent=1)
            tally[cs] += 1
            print(f'{pid}/{cid:4s} -> {str(cs):14s} ' +
                  ' '.join(f"{m.split('-')[0]}:{per_model[m].get('state')}" for m in MODELS), flush=True)
    print('REJUDGE-DONE', dict(tally), flush=True)


if __name__ == '__main__':
    main()
