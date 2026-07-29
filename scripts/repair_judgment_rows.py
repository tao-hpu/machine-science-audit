#!/usr/bin/env python3
"""修复判定目录里的残行:重调当时 API 失败的裁判,按 judge_batch 同一 facet 级共识重导 state。

目标行 = per_model 里任一裁判带 error,或 consensus_state_v2 ∈ {null, split}。
只动残行,健康行零改写;幂等可重跑。

用法:
  python3 scripts/repair_judgment_rows.py --out data/judgments_human_rescue \
      --ext data/extractions_human --nbr data/neighbors_human_rescue [--dry]
"""
import argparse, json, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)
import scripts.judge_v3 as J
from scripts.facet_consensus import covset, derive_state, TIEBREAK

PRIMARY = ['gpt-4o', 'claude-sonnet-4-6']


def rejudge(model, contrib, nbrs, pm):
    try:
        r = J.judge_one(model, contrib, nbrs, keep_raw=True)
        pm[model] = {'state': r['state'], 'facets': r['facets'],
                     'self_check': r['self_check_triggered'], 'raw': r['raw']}
    except Exception as ex:
        pm[model] = {'state': None, 'facets': None, 'error': repr(ex)[:100]}
    time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--ext', required=True)
    ap.add_argument('--nbr', required=True)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    out, ext_dir, nbr_dir = map(pathlib.Path, (a.out, a.ext, a.nbr))
    fixed = still_bad = 0
    for f in sorted(out.glob('*.json')):
        if f.name.startswith('_'):
            continue
        j = json.load(open(f))
        pid = j['paper_id']
        changed = False
        for c in j['contributions']:
            pm = c.get('per_model') or {}
            broken = [m for m, v in pm.items() if v.get('error')]
            if not broken and c.get('consensus_state_v2') not in (None, 'split'):
                continue
            if a.dry:
                print(f"{pid}/{c['cid']} state={c.get('consensus_state_v2')} broken={broken}")
                continue
            ext = json.load(open(ext_dir / f'{pid}.json'))
            nbrj = json.load(open(nbr_dir / f'{pid}.json'))
            ec = next(x for x in ext['contributions'] if x['id'] == c['cid'])
            nc = next(x for x in nbrj['contributions'] if x['id'] == c['cid'])
            nbrs = nc['neighbors']
            for m in broken:
                rejudge(m, ec, nbrs, pm)
            # 主裁判齐了但存在 facet 分歧且还没有第三票 → 补仲裁
            if all(pm.get(m, {}).get('facets') for m in PRIMARY):
                disagreed = [fa for fa in J.FACETS
                             if bool(covset(pm[PRIMARY[0]]['facets'], fa))
                             != bool(covset(pm[PRIMARY[1]]['facets'], fa))]
                if disagreed and not pm.get(TIEBREAK, {}).get('facets'):
                    rejudge(TIEBREAK, ec, nbrs, pm)
            # facet 级共识重导(与 judge_batch 同逻辑)
            voters = [m for m in (PRIMARY + [TIEBREAK]) if pm.get(m, {}).get('facets')]
            if len(voters) >= 2:
                cons, unresolved = {}, []
                for fa in J.FACETS:
                    votes = [covset(pm[m]['facets'], fa) for m in voters]
                    covered = [v for v in votes if v]
                    if len(covered) * 2 > len(votes):
                        cons[fa] = set().union(*covered)
                    elif len(covered) * 2 < len(votes):
                        cons[fa] = set()
                    else:
                        unresolved.append(fa); cons[fa] = set()
                c['consensus_state_v2'] = 'split' if unresolved else derive_state(cons)
                if unresolved:
                    c['consensus_unresolved'] = unresolved
                else:
                    c.pop('consensus_unresolved', None)
                c['consensus_facets'] = {fa: sorted(cons[fa]) if cons[fa] else None for fa in J.FACETS}
            status = c.get('consensus_state_v2')
            if status in (None, 'split') or any(v.get('error') for v in pm.values()):
                still_bad += 1
            else:
                fixed += 1
            changed = True
            print(f"{pid}/{c['cid']} -> {status} " +
                  ' '.join(f"{m.split('-')[0]}:{pm[m].get('state')}" for m in pm), flush=True)
        if changed and not a.dry:
            json.dump(j, open(f, 'w'), ensure_ascii=False, indent=1)
    print(f'REPAIR-DONE fixed={fixed} still_bad={still_bad}')


if __name__ == '__main__':
    main()
