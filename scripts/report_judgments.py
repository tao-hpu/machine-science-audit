#!/usr/bin/env python3
"""判定结果汇报/sanity 脚本:三态分布、novel 驱动 facet、仲裁率、错误率,可多目录对比。

预写于 2026-07-08(等 OA 索引的空档),P2/P4/P6 判定一落地直接出数。
- 三态取 consensus_state_v2(v7 协议的确定性共识);
- novel 驱动 facet = facet-novel 行里共识为「未覆盖」(None)的 facet(防 mirage 检查:domain 应≈0);
- 人类侧若给 --decisions(matched_166.json),按 accept/reject 分层。

用法:
  python3 scripts/report_judgments.py data/judgments_full
  python3 scripts/report_judgments.py data/judgments_full data/judgments_human   # 并排对比
  python3 scripts/report_judgments.py data/judgments_human --decisions data/human_iclr2025/matched_166.json
"""
import argparse, json, pathlib, sys
from collections import Counter

FACETS = ('purpose', 'mechanism', 'evaluation', 'domain')


def load_rows(d):
    rows = []
    for f in sorted(pathlib.Path(d).glob('*.json')):
        if f.name.startswith('_'):
            continue
        try:
            j = json.load(open(f))
        except Exception:
            continue
        for r in j.get('contributions', []):
            r['_pid'] = j.get('paper_id', f.stem)
            rows.append(r)
    return rows


def summarize(rows, label):
    n = len(rows)
    st = Counter(r.get('consensus_state_v2') or 'ERROR/null' for r in rows)
    arb = sum(1 for r in rows if 'gemini-2.5-pro' in (r.get('per_model') or {}))
    err = sum(1 for r in rows if any(v.get('error') for v in (r.get('per_model') or {}).values()))
    drivers = Counter()
    for r in rows:
        if r.get('consensus_state_v2') == 'facet-novel':
            for fa, cov in (r.get('consensus_facets') or {}).items():
                if cov is None:
                    drivers[fa] += 1
    types = Counter(r.get('type') or '?' for r in rows)
    print(f'== {label}  (n={n})')
    if not n:
        return
    for s, c in st.most_common():
        print(f'   {s:16s} {c:4d}  ({100*c/n:.1f}%)')
    print(f'   仲裁率 {arb}/{n} ({100*arb/max(1,n):.0f}%) | error行 {err} ({100*err/max(1,n):.1f}%)')
    print(f'   novel 驱动 facet: ' + ', '.join(f'{fa}:{drivers.get(fa,0)}' for fa in FACETS) +
          ('   ⚠️ domain≠0,查 mirage!' if drivers.get('domain') else ''))
    print(f'   贡献类型: {dict(types)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--decisions', help='matched_166.json:人类侧按 accept/reject 分层')
    a = ap.parse_args()
    for d in a.dirs:
        rows = load_rows(d)
        summarize(rows, d)
        if a.decisions:
            dec = {}
            for r in json.load(open(a.decisions)):
                v = json.dumps(r.get('decision') or r.get('content', {}).get('venue', '')).lower()
                dec[r['id']] = 'accept' if ('accept' in v or 'poster' in v or 'oral' in v or 'spotlight' in v) else 'reject'
            for grp in ('accept', 'reject'):
                summarize([r for r in rows if dec.get(r['_pid']) == grp], f'{d} [{grp}]')
        print()


if __name__ == '__main__':
    main()
