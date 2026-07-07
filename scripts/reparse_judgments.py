#!/usr/bin/env python3
"""从已落盘的 raw stage1/stage2 离线重算判定(parser 修复后免重跑 LLM)。
用法: python3 scripts/reparse_judgments.py data/judgments_facet_val
"""
import json, glob, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scripts.judge_v3 as J
from collections import Counter


def recompute(raw):
    verdicts = J.parse_verdicts(raw.get('stage1', ''))
    none = [f for f, v in verdicts.items() if v is None]
    for fac, v in J.parse_stage2(raw.get('stage2', '')).items():
        if fac in none:
            verdicts[fac] = v
    fac = {f: {'covered_by': (None if verdicts[f] == 'MISSING' else verdicts[f])} for f in J.FACETS}
    if any(v == 'MISSING' for v in verdicts.values()):
        state, spa = None, None
    else:
        sets = [set(v if isinstance(v, list) else [v]) if v is not None else set() for v in verdicts.values()]
        common = set.intersection(*sets) if all(sets) else set()
        spa = min(common) if common else None
        state = 'covered' if spa else ('recombination' if all(sets) else 'facet-novel')
    fac['single_paper_covers_all'] = spa
    return state, fac, none


def consensus(states):
    v = [s for s in states if s]
    if not v: return None
    top = Counter(v).most_common()
    return 'split' if len(top) > 1 and top[0][1] == top[1][1] else top[0][0]


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else 'data/judgments_facet_val'
    changed_n = 0
    for f in sorted(glob.glob(f'{target}/FA*.json')):
        j = json.load(open(f))
        changed = False
        for c in j['contributions']:
            states = []
            for m, r in c['per_model'].items():
                if not r.get('raw'):
                    states.append(r.get('state')); continue
                state, fac, none = recompute(r['raw'])
                if state != r.get('state') or fac != r.get('facets'):
                    changed = True; changed_n += 1
                    print(f"{j['paper_id']}/{c['cid']} {m}: {r.get('state')} -> {state}")
                r['state'], r['facets'], r['self_check'] = state, fac, none
                states.append(state)
            cs = consensus(states)
            if cs != c.get('consensus_state'):
                changed = True
                print(f"{j['paper_id']}/{c['cid']} consensus: {c.get('consensus_state')} -> {cs}")
                c['consensus_state'] = cs
        if changed:
            json.dump(j, open(f, 'w'), ensure_ascii=False, indent=1)
    print('REPARSE-DONE changed_models=', changed_n)


if __name__ == '__main__':
    main()
