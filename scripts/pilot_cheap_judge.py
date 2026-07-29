#!/usr/bin/env python3
"""降档试点:gpt-4o-mini 单裁判重判 20 条验证批,对照 v7 定案共识算 facet 级一致率。

背景(2026-07-08,Tao 问能否降成本):判定总账 ~$100-145,A4S 占约一半。
若 mini 对金标准(data/judgments_v7_val/ 的 consensus_facets,v7+facet 共识+仲裁)
的 facet 覆盖一致率 ≥95%,则仅在 A4S 语料把 gpt-4o 降为 gpt-4o-mini(FARS/人类保持原配);
低于就维持原配,把本试点记进 research-log 当"降档已考察并否决"的证据。

口径:比"facet 是否被覆盖"(bool),与协议的分歧定义一致;引证具体编号不比。
"""
import json, glob, sys, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import scripts.judge_v3 as J
from scripts.facet_consensus import covset

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'gpt-4o-mini'
OUT = f'data/pilot_cheap_judge_{MODEL.replace("/", "_")}.json'

res = json.load(open(OUT)) if os.path.exists(OUT) else {}
gold_rows = []
for f in sorted(glob.glob('data/judgments_v7_val/FA*.json')):
    d = json.load(open(f))
    pid = d['paper_id'] if isinstance(d, dict) else f.split('/')[-1][:-5]
    for r in (d['contributions'] if isinstance(d, dict) else d):
        if r.get('consensus_facets') is not None:
            gold_rows.append((pid, r['cid'], r))

ext_cache, nbr_cache = {}, {}
for pid, cid, gr in gold_rows:
    key = f'{pid}/{cid}'
    if key in res:
        continue
    if pid not in ext_cache:
        ext_cache[pid] = {c['id']: c for c in json.load(open(f'data/extractions/{pid}.json'))['contributions']}
        nbr_cache[pid] = {c['id']: c['neighbors'] for c in json.load(open(f'data/neighbors_facet_val/{pid}.json'))['contributions']}
    try:
        r = J.judge_one(MODEL, ext_cache[pid][cid], nbr_cache[pid][cid], keep_raw=True)
        res[key] = {'state': r['state'], 'facets': r['facets']}
    except Exception as e:
        res[key] = {'state': None, 'facets': None, 'error': repr(e)[:120]}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(key, '->', res[key].get('state'), flush=True)
    time.sleep(0.3)

agree = tot = 0
diffs = []
for pid, cid, gr in gold_rows:
    key = f'{pid}/{cid}'
    m = res.get(key) or {}
    if not m.get('facets'):
        continue
    for fa in J.FACETS:
        gold_cov = gr['consensus_facets'].get(fa) is not None
        mini_cov = bool(covset(m['facets'], fa))
        tot += 1
        if gold_cov == mini_cov:
            agree += 1
        else:
            diffs.append(f'{key}/{fa}: gold={"cov" if gold_cov else "none"} {MODEL}={"cov" if mini_cov else "none"}')
print(f'\n{MODEL} vs v7 gold: facet 覆盖一致率 {agree}/{tot} ({100*agree/max(1,tot):.1f}%)')
for d in diffs:
    print(' ', d)
