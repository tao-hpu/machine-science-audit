#!/usr/bin/env python3
"""Camera-ready robustness analyses on the frozen judgments (no new model calls).

Answers the workshop reviewers' questions that the frozen data can settle:
  1. claim-type stratification and direct standardization;
  2. how many facet-novel verdicts rest on the purpose facet alone, and the
     machine-vs-human gap under stricter derivation rules;
  3. the gap after dropping weakly matched pairs;
  4. search failure vs. judge failure on the gold prior-art sample;
  5. pair recall if cross-contribution misallocation were fixed.

Usage: python3 scripts/cr_robustness.py [--out data/cr_robustness.json]
"""
import argparse
import ast
import json
import pathlib
import random
from collections import Counter

ARMS = {'machine': 'data/judgments_fars_rescue', 'human': 'data/judgments_human_rescue'}
FACETS = ('purpose', 'mechanism', 'evaluation', 'domain')
STATES = ('facet-novel', 'recombination', 'covered')


def load(d):
    rows = []
    for f in sorted(pathlib.Path(d).glob('*.json')):
        if f.name.startswith('_') or f.name.startswith('.'):
            continue
        j = json.load(open(f))
        for r in j.get('contributions', []):
            cf = r.get('consensus_facets')
            if isinstance(cf, str):
                cf = ast.literal_eval(cf)
            rows.append({'pid': j.get('paper_id', f.stem), 'cid': r['cid'],
                         'type': r.get('type'), 'state': r['consensus_state_v2'],
                         'uncovered': [fa for fa in FACETS if fa in (cf or {}) and cf[fa] is None]})
    return rows


def rate(rows, pred=lambda r: r['state'] == 'facet-novel'):
    return sum(1 for r in rows if pred(r)) / len(rows) if rows else float('nan')


def boot_gap(rows, match, pred, n_boot=10000, seed=42):
    """95% CI of the machine-minus-human gap, resampling matched pairs (the unit of design)."""
    by_pid = {}
    for arm, rs in rows.items():
        for r in rs:
            by_pid.setdefault(r['pid'], []).append(r)
    pairs = [(m['matched_to'], m['id']) for m in match]
    rng = random.Random(seed)
    gaps = []
    for _ in range(n_boot):
        mm, hh = [], []
        for _ in pairs:
            mp, hp = pairs[rng.randrange(len(pairs))]
            mm += by_pid.get(mp, [])
            hh += by_pid.get(hp, [])
        gaps.append(rate(mm, pred) - rate(hh, pred))
    gaps.sort()
    return [round(100 * gaps[int(0.025 * n_boot)], 1), round(100 * gaps[int(0.975 * n_boot)], 1)]


def pct(x):
    return round(100 * x, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/cr_robustness.json')
    a = ap.parse_args()
    rows = {arm: load(d) for arm, d in ARMS.items()}
    out = {'n': {arm: len(r) for arm, r in rows.items()}}
    out['facet_novel'] = {arm: pct(rate(r)) for arm, r in rows.items()}

    # 1. claim type
    strat = {}
    for arm, rs in rows.items():
        for t in ('method', 'finding'):
            sub = [r for r in rs if r['type'] == t]
            c = Counter(r['state'] for r in sub)
            strat[f'{arm}/{t}'] = {'n': len(sub), **{s: pct(c[s] / len(sub)) for s in STATES}}
    mix = {arm: Counter(r['type'] for r in rs if r['type'] in ('method', 'finding'))
           for arm, rs in rows.items()}

    def standardized(arm, to):
        tot = sum(mix[to].values())
        return sum(mix[to][t] / tot * rate([r for r in rows[arm] if r['type'] == t])
                   for t in ('method', 'finding'))
    out['type_stratified'] = strat
    out['type_standardized'] = {'machine_at_human_mix': round(100 * standardized('machine', 'human'), 2),
                                'human_at_machine_mix': round(100 * standardized('human', 'machine'), 2)}

    # 2. purpose-only verdicts and stricter derivation rules
    rules = {
        'any_uncovered (frozen rule)': lambda r: r['state'] == 'facet-novel',
        'non-purpose facet uncovered': lambda r: r['state'] == 'facet-novel'
            and any(f != 'purpose' for f in r['uncovered']),
        'mechanism uncovered': lambda r: r['state'] == 'facet-novel' and 'mechanism' in r['uncovered'],
        '>=2 facets uncovered': lambda r: r['state'] == 'facet-novel' and len(r['uncovered']) >= 2,
    }
    out['purpose_only'] = {}
    for arm, rs in rows.items():
        nov = [r for r in rs if r['state'] == 'facet-novel']
        po = sum(1 for r in nov if r['uncovered'] == ['purpose'])
        out['purpose_only'][arm] = {'facet_novel': len(nov), 'purpose_only': po,
                                    'share': pct(po / len(nov))}
    match = json.load(open('data/human_iclr2025/matched_166.json'))
    out['stricter_rules'] = {
        name: {arm: pct(rate(rs, f)) for arm, rs in rows.items()} | {
            'gap_pts': round(100 * (rate(rows['machine'], f) - rate(rows['human'], f)), 1),
            'gap_ci95': boot_gap(rows, match, f)}
        for name, f in rules.items()}

    # 3. drop weakly matched pairs
    sims = sorted(float(m['sim']) for m in match)
    cuts = {'all pairs': 0.0,
            'sim >= median': sims[len(sims) // 2],
            'sim >= upper quartile': sims[3 * len(sims) // 4]}
    out['match_similarity'] = {'min': sims[0], 'median': sims[len(sims) // 2],
                               'q3': sims[3 * len(sims) // 4], 'max': sims[-1]}
    out['match_filter'] = {}
    for name, c in cuts.items():
        keep = [m for m in match if float(m['sim']) >= c]
        hp = {m['id'] for m in keep}
        mp = {m['matched_to'] for m in keep}
        m_rows = [r for r in rows['machine'] if r['pid'] in mp]
        h_rows = [r for r in rows['human'] if r['pid'] in hp]
        out['match_filter'][name] = {
            'threshold': round(c, 4), 'pairs': len(keep),
            'machine': {'n': len(m_rows), 'facet_novel': pct(rate(m_rows))},
            'human': {'n': len(h_rows), 'facet_novel': pct(rate(h_rows))},
            'gap_pts': round(100 * (rate(m_rows) - rate(h_rows)), 1),
            'gap_ci95': boot_gap({'machine': m_rows, 'human': h_rows}, keep, rules['any_uncovered (frozen rule)'])}

    # 4 + 5. gold sample: search failure vs judge failure; misallocation
    rep = json.load(open('data/v1_gold/recall_report.json'))
    sample = json.load(open('data/v1_gold/sample.json'))
    state_of = {(arm, s['paper_id'], s['cid']): s['state']
                for arm, lst in sample['arms'].items() for s in lst}
    by_c = {}
    for d in rep['detail']:
        k = (d['arm'], d['paper_id'], d['cid'])
        by_c.setdefault(k, []).append(d)
    gold = {}
    for arm in ('machine', 'human'):
        tab = {}
        for k, ds in by_c.items():
            if k[0] != arm:
                continue
            st = state_of.get(k)
            if st is None:
                # hand-seeded contributions are not in the stratified sample; read their state
                st = next((r['state'] for r in rows[arm] if (r['pid'], r['cid']) == k[1:]), '?')
            hit = any(d['hit_rank'] is not None for d in ds)
            tab.setdefault(st, Counter())['retrieved' if hit else 'missed'] += 1
            if st == 'facet-novel' and hit:
                unc = next((r['uncovered'] for r in rows[arm] if (r['pid'], r['cid']) == k[1:]), [])
                if any(d['hit_rank'] is not None and d['facet'] in unc for d in ds):
                    # a gold paper tagged with the very facet the panel called uncovered was in view
                    tab[st]['retrieved_same_facet'] += 1
        pairs = [d for d in rep['detail'] if d['arm'] == arm]
        direct = sum(1 for d in pairs if d['hit_rank'] is not None)
        misalloc = sum(1 for d in pairs if d['hit_rank'] is None and d['paper_level_hit'])
        gold[arm] = {'by_state': {s: dict(c) for s, c in tab.items()},
                     'pairs': len(pairs), 'pair_recall': pct(direct / len(pairs)),
                     'misallocated': misalloc,
                     'pair_recall_if_shared': pct((direct + misalloc) / len(pairs))}
    out['gold'] = gold

    json.dump(out, open(a.out, 'w'), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
