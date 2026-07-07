#!/usr/bin/env python3
"""Facet 级共识 + 第三裁判仲裁(尺子 v7 管线第⑤点)。

背景:state 级数票是脆弱聚合——recombination 要求 4 facet 全覆盖,单 facet 分歧
在边界上翻整个 state,且存在「state 同、判据 facet 不同」的伪一致。
本脚本把共识降到 facet 级:
  1. 两主裁判(gpt-4o / claude-sonnet-4-6)对某 facet 的覆盖判断一致 → 直接采纳;
  2. 不一致 → 调第三裁判(gemini-2.5-pro,BIG_BUDGET)对该贡献全量判定,facet 级 2/3 多数票;
  3. state 由共识 facet 确定性导出:
     - 任一共识 facet 未覆盖 → facet-novel
     - 全覆盖且各 facet「胜方引证集合」的交集非空 → covered(单篇覆盖全部)
     - 全覆盖无单篇交集 → recombination
仲裁结果写回同一 JSON(per_model 增第三家,新增 consensus_facets / consensus_state_v2)。

用法: python3 scripts/facet_consensus.py data/judgments_v7_val [--dry]
"""
import json, glob, sys, time, pathlib
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scripts.judge_v3 as J

TIEBREAK = 'gemini-2.5-pro'
PRIMARY = ['gpt-4o', 'claude-sonnet-4-6']


def covset(facets, fa):
    v = (facets or {}).get(fa)
    v = v.get('covered_by') if isinstance(v, dict) else v
    if v is None: return set()
    return set(v) if isinstance(v, list) else {v}


def derive_state(cons):
    """cons: {facet: set(引证) 或 空 set=未覆盖}"""
    if any(not s for s in cons.values()):
        return 'facet-novel'
    common = set.intersection(*cons.values())
    return 'covered' if common else 'recombination'


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    dry = '--dry' in sys.argv
    target = args[0] if args else 'data/judgments_v7_val'
    tally = Counter(); tb_calls = 0
    for f in sorted(glob.glob(f'{target}/FA*.json')):
        j = json.load(open(f))
        ext = json.load(open(f"data/extractions/{j['paper_id']}.json"))
        nbr = json.load(open(f"data/neighbors_facet_val/{j['paper_id']}.json"))
        changed = False
        for c in j['contributions']:
            pm = c['per_model']
            if not all(m in pm and pm[m].get('facets') for m in PRIMARY):
                continue
            disagreed = [fa for fa in J.FACETS
                         if bool(covset(pm[PRIMARY[0]]['facets'], fa)) != bool(covset(pm[PRIMARY[1]]['facets'], fa))]
            if disagreed and TIEBREAK not in pm:
                if dry:
                    print(f"{j['paper_id']}/{c['cid']} 需仲裁: {disagreed}")
                    continue
                ec = next(x for x in ext['contributions'] if x['id'] == c['cid'])
                nc = next(x for x in nbr['contributions'] if x['id'] == c['cid'])
                try:
                    r = J.judge_one(TIEBREAK, ec, nc['neighbors'], keep_raw=True)
                    pm[TIEBREAK] = {'state': r['state'], 'facets': r['facets'],
                                    'self_check': r['self_check_triggered'], 'raw': r['raw']}
                    tb_calls += 1; changed = True
                except Exception as ex:
                    pm[TIEBREAK] = {'state': None, 'facets': None, 'error': repr(ex)[:120]}
                    changed = True
                time.sleep(0.5)
            if dry: continue
            # facet 级共识:一致直接采纳;分歧看第三票(缺失/失败则记未覆盖一票不算,按主裁判平票保守走 gpt∪sonnet 中覆盖方? 不——平票无解时记 unresolved)
            cons = {}; unresolved = []
            for fa in J.FACETS:
                votes = []
                for m in (PRIMARY + ([TIEBREAK] if TIEBREAK in pm and pm[TIEBREAK].get('facets') else [])):
                    votes.append(covset(pm[m]['facets'], fa))
                covered_votes = [v for v in votes if v]
                if len(covered_votes) * 2 > len(votes):        # 多数覆盖
                    cons[fa] = set().union(*covered_votes)
                elif len(covered_votes) * 2 < len(votes):      # 多数未覆盖
                    cons[fa] = set()
                else:                                          # 平票(第三票缺失)
                    unresolved.append(fa); cons[fa] = set()
            if unresolved:
                c['consensus_state_v2'] = 'split'
                c['consensus_unresolved'] = unresolved
            else:
                c['consensus_state_v2'] = derive_state(cons)
                c.pop('consensus_unresolved', None)
            c['consensus_facets'] = {fa: sorted(cons[fa]) if cons[fa] else None for fa in J.FACETS}
            tally[c['consensus_state_v2']] += 1
            changed = True
        if changed and not dry:
            json.dump(j, open(f, 'w'), ensure_ascii=False, indent=1)
    print(f'CONSENSUS-DONE tiebreak_calls={tb_calls} states={dict(tally)}')


if __name__ == '__main__':
    main()
