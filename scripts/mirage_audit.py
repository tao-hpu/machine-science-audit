#!/usr/bin/env python3
"""Mirage 对抗审计(可复现重建版):对判为 facet-novel 的贡献,让 gpt-4o 扮演敌意审计员,
从同一份近邻列表里硬找 purpose 覆盖。找到(REFUTED)= 该 novel 判定是 mirage 嫌疑;
找不到(HOLDS)= 判定在对抗压力下站得住。

原版 2026-07-13 在会话 scratchpad 跑过(机器 50% / 人类 40% 被反驳)但脚本与输出均未入仓;
本版 seed 固定、逐条落盘,论文引用以本版输出为准。

用法:
  python3 scripts/mirage_audit.py                # 两臂各抽 20,seed 42
  python3 scripts/mirage_audit.py --k 20 --seed 42 --out data/mirage_audit.json
"""
import argparse, json, pathlib, random, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)
import scripts.judge_v3 as J

ARMS = {
    'machine': ('data/judgments_fars_rescue', 'data/extractions', 'data/neighbors_fars_cito_rescue'),
    'human':   ('data/judgments_human_rescue', 'data/extractions_human', 'data/neighbors_human_rescue'),
}
AUDITOR = 'gpt-4o'

PROMPT = """You are an adversarial auditor in a prior-art coverage pipeline. A judge panel \
concluded that the PURPOSE of the contribution below has NO coverage in the retrieved neighbor \
list. Your only job is to try as hard as possible to REFUTE that conclusion: find any neighbor \
whose work pursues the same purpose.

Coverage standard (same as the panel's): for method-type contributions, a neighbor covers the \
purpose if it pursues the same objective at the problem-class level (host architecture, dataset, \
or wording may differ). For finding-type contributions, a neighbor covers the purpose if it \
investigates the same empirical relationship (same variables and direction; vocabulary may \
differ). Sharing only a generic tool, a broad research area, or the same problem space with a \
different objective does NOT count as coverage.

Contribution (type: {ctype}):
- purpose: {purpose}
- mechanism: {mechanism}
- evaluation: {evaluation}
- domain: {domain}

Neighbors:
{papers}

Rules: cite evidence verbatim from a neighbor's title/abstract. If, after honest effort, no \
neighbor meets the standard, you must concede HOLDS. Do not stretch analogies to force a refutation.

Output exactly:
VERDICT: REFUTED or HOLDS
NEIGHBOR: [index] (only if REFUTED)
EVIDENCE: verbatim quote + one-sentence justification (only if REFUTED)"""

# harsh 变体:去掉「按 panel 标准 + 不许硬拉类比」护栏,贴近 2026-07-13 原版的裸敌意口径,
# 用于量化审计工具自身的 prompt 敏感性(strict vs harsh 的差 = 工具不确定度)。
PROMPT_HARSH = """You are a hostile reviewer. A judge panel concluded that the PURPOSE of the \
contribution below is genuinely novel, with no coverage in the retrieved neighbor list. You \
believe the panel is too generous. Scrutinize the neighbor list and find any paper that already \
pursues this purpose, even under different terminology.

Contribution (type: {ctype}):
- purpose: {purpose}
- mechanism: {mechanism}
- evaluation: {evaluation}
- domain: {domain}

Neighbors:
{papers}

Output exactly:
VERDICT: REFUTED or HOLDS
NEIGHBOR: [index] (only if REFUTED)
EVIDENCE: verbatim quote + one-sentence justification (only if REFUTED)"""


def novel_purpose_rows(jdir):
    rows = []
    for f in sorted(pathlib.Path(jdir).glob('*.json')):
        if f.name.startswith('_'):
            continue
        j = json.load(open(f))
        for c in j['contributions']:
            if c.get('consensus_state_v2') != 'facet-novel':
                continue
            if (c.get('consensus_facets') or {}).get('purpose') is None:
                rows.append((j['paper_id'], c['cid']))
    return rows


def get_contrib(ext, pid, cid):
    for c in json.load(open(f'{ext}/{pid}.json'))['contributions']:
        if c['id'] == cid:
            return c


def get_neighbors(nbr, pid, cid):
    for c in json.load(open(f'{nbr}/{pid}.json'))['contributions']:
        if c['id'] == cid:
            return c.get('neighbors', [])
    return []


def audit_one(contrib, neighbors, prompt=PROMPT):
    papers = '\n'.join(f"[{i+1}] ({n.get('date')}) {n['title']} — {(n.get('abstract') or '')[:500]}"
                       for i, n in enumerate(neighbors))
    raw = J.chat(AUDITOR, prompt.format(papers=papers, ctype=contrib.get('type', 'method'),
                                        **{k: contrib[k] for k in ('purpose', 'mechanism', 'evaluation', 'domain')}),
                 max_tokens=1200)
    verdict = 'REFUTED' if 'VERDICT: REFUTED' in raw.upper().replace('**', '') else \
              ('HOLDS' if 'HOLDS' in raw.upper() else 'PARSE-FAIL')
    return verdict, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=20, help='每臂抽样条数')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--variant', choices=['strict', 'harsh'], default='strict',
                    help='strict=对齐 panel 标准+反拉伸护栏;harsh=裸敌意(贴近原版)')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    prompt = PROMPT if a.variant == 'strict' else PROMPT_HARSH
    out_path = pathlib.Path(a.out or ('data/mirage_audit.json' if a.variant == 'strict'
                                      else f'data/mirage_audit_{a.variant}.json'))
    results = json.load(open(out_path)) if out_path.exists() else {'meta': {}, 'items': []}
    done = {(r['arm'], r['pid'], r['cid']) for r in results['items']}
    results['meta'] = {'auditor': AUDITOR, 'k': a.k, 'seed': a.seed, 'variant': a.variant,
                       'facet': 'purpose', 'arms': {k: v[0] for k, v in ARMS.items()}}
    for arm, (jdir, ext, nbr) in ARMS.items():
        pool = novel_purpose_rows(jdir)
        rng = random.Random(a.seed)
        sample = rng.sample(pool, min(a.k, len(pool)))
        print(f'== {arm}: pool={len(pool)} sampled={len(sample)}')
        for pid, cid in sample:
            if (arm, pid, cid) in done:
                continue
            c = get_contrib(ext, pid, cid)
            nbrs = get_neighbors(nbr, pid, cid)
            verdict, raw = audit_one(c, nbrs, prompt)
            results['items'].append({'arm': arm, 'pid': pid, 'cid': cid,
                                     'verdict': verdict, 'raw': raw})
            json.dump(results, open(out_path, 'w'), ensure_ascii=False, indent=1)
            print(f'{arm} {pid}/{cid} -> {verdict}', flush=True)
            time.sleep(0.5)
    # 汇总
    print('\n== SUMMARY')
    for arm in ARMS:
        items = [r for r in results['items'] if r['arm'] == arm]
        n = len(items)
        ref = sum(1 for r in items if r['verdict'] == 'REFUTED')
        pf = sum(1 for r in items if r['verdict'] == 'PARSE-FAIL')
        print(f'{arm}: n={n} refuted={ref} ({ref/n:.0%}) holds={n-ref-pf} parse-fail={pf}')


if __name__ == '__main__':
    main()
