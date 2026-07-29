#!/usr/bin/env python3
"""练习题包(标注前热身对齐口径):~10 条揭示答案的案例 + 指南 4 个教学例。
选案纪律:covered/recombination 取两主裁判「无争议」清晰案例;facet-novel 只取
「通过对抗审计(holds)」的——不拿 mirage 当标准答案。两臂都取。
用法:python3 scripts/build_practice_items.py --out data/annotation"""
import argparse, json, glob, os, html, hashlib

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARMS = {'machine': ('data/judgments_fars_rescue', 'data/extractions', 'data/neighbors_fars_cito_rescue'),
        'human':   ('data/judgments_human_rescue', 'data/extractions_human', 'data/neighbors_human_rescue')}
FACETS = ('purpose', 'mechanism', 'evaluation', 'domain')
# facet-novel 案例只用对抗审计 holds 的(mirage_audit.py 输出里 machine 站得住的)
HOLDS_FN = [('machine', 'FA0031', 'C1'), ('machine', 'FA0052', 'C1'),
            ('machine', 'FA0082', 'C3'), ('machine', 'FA0201', 'C1')]


def load_contrib(ext, pid, cid):
    for c in json.load(open(f'{ext}/{pid}.json'))['contributions']:
        if c['id'] == cid:
            return c


def load_neighbors(nbr, pid, cid):
    for c in json.load(open(f'{nbr}/{pid}.json'))['contributions']:
        if c['id'] == cid:
            return c.get('neighbors', [])
    return []


def agreement_cases(state, k):
    picked = []
    for arm, (jdir, _, _) in ARMS.items():
        for f in sorted(glob.glob(f'{jdir}/*.json')):
            d = json.load(open(f)); pid = d['paper_id']
            for c in d['contributions']:
                pm = c.get('per_model', {})
                sts = [m.get('state') for m in pm.values() if isinstance(m, dict)]
                cs = c.get('consensus_state_v2') or c.get('consensus_state')
                cf = c.get('consensus_facets') or {}
                if len(sts) >= 2 and len(set(sts)) == 1 and sts[0] == cs == state:
                    picked.append((arm, pid, c['cid'], cs, cf))
    step = max(1, len(picked) // max(1, k))
    return picked[::step][:k]


def enrich(arm, pid, cid, state, cf=None):
    _, ext, nbr = ARMS[arm]
    c = load_contrib(ext, pid, cid); nb = load_neighbors(nbr, pid, cid)
    if cf is None:
        for f in glob.glob(f'{ARMS[arm][0]}/{pid}.json'):
            for cc in json.load(open(f))['contributions']:
                if cc['cid'] == cid:
                    cf = cc.get('consensus_facets') or {}
    # 答案说明:哪些 facet 有覆盖(paper#)/哪些无(novel)
    fa = {k: (('covered by #' + str(cf.get(k))) if cf.get(k) not in (None, [], 'null') else 'NONE (novel)') for k in FACETS}
    return {'state': state, 'type': c.get('type', ''),
            'facets': {k: (c.get(k) or '').strip() for k in FACETS},
            'answer_facets': fa,
            'neighbors': [{'n': j + 1, 'title': h['title'], 'abstract': (h.get('abstract') or '')[:400]}
                          for j, h in enumerate(nb[:15])]}


def build(outdir):
    os.makedirs(outdir, exist_ok=True)
    items = []
    for arm, pid, cid in HOLDS_FN:
        items.append(enrich(arm, pid, cid, 'facet-novel'))
    for arm, pid, cid, st, cf in agreement_cases('covered', 3):
        items.append(enrich(arm, pid, cid, st, cf))
    for arm, pid, cid, st, cf in agreement_cases('recombination', 3):
        items.append(enrich(arm, pid, cid, st, cf))
    path = os.path.join(outdir, 'practice_items.html')
    open(path, 'w').write(render(items))
    print(f'✅ 练习题 {len(items)} 条 → {path}')


def render(items):
    esc = html.escape
    rows = ''
    for i, it in enumerate(items, 1):
        fh = ''.join(f'<div class="facet"><b>{k}</b> {esc(v)}</div>' for k, v in it['facets'].items() if v)
        nh = ''.join(f'<div class="nbr"><span class="num">[{n["n"]}]</span>{esc(n["title"])}. <span class=ab>{esc(n["abstract"])}</span></div>' for n in it['neighbors'])
        ans = ' · '.join(f'{k[0].upper()}:{esc(v)}' for k, v in it['answer_facets'].items())
        rows += f'''<div class="card"><div class="qn">Practice {i} · type: {esc(it["type"])}</div>{fh}
<div class="nbrs"><b>Prior papers:</b>{nh}</div>
<details class="ans"><summary>Show reference answer</summary>
<p><b>State: {esc(it["state"])}</b></p><p class=fa>Per-facet: {ans}</p>
<p class=hint>Try to reach this yourself first. If you disagree, that's exactly what the calibration discussion is for.</p></details></div>'''
    guide = '''<div class="teach"><h3>Worked teaching examples (read first)</h3>
<p><b>1 — Cross-domain Recombination (easy to over-call as Facet-novel).</b> A paper claims a novel "benign-padding attack": prepend benign text to harmful content so the safety classifier's FNR jumps 0.05→1.0. Looks like a new mechanism — but a neighbor is "Good Word Attacks on Statistical Spam Filters" (2005): pad spam with benign words to dilute the statistical signal. Mechanism is substantively equivalent (dilution evasion); only the domain (LLM safety) is new → <b>Recombination</b>. Lesson: abstract the mechanism to the attack primitive, don't be fooled by the application skin.</p>
<p><b>2 — Signal-substitution Recombination.</b> "Replace FOREVER's parameter-space replay signal with function-space KL divergence." KL regularization already exists in neighbors (FROMP); the replay-scheduling purpose exists in a neighbor (FOREVER); no single paper covers both → <b>Recombination</b>.</p>
<p><b>3 — Finding-type contribution.</b> "We find KL-drift and parameter-norm correlate >0.97 within-task but diverge 14.24% at task boundaries." Fit the four facets: mechanism (correlation analysis) & evaluation (standard benchmark) are generic designs → covered if any neighbor used them; the real question is <b>purpose</b> — does any neighbor report this specific relationship? If not, novelty lands on purpose.</p>
<p><b>4 — Recombination subtypes (optional label).</b> domain-transfer / element-assembly / substitution. If unsure, leave blank.</p></div>'''
    return f'''<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Annotation Practice</title><style>
body{{margin:0;font:15px/1.55 -apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#faf9f7;color:#1a1a1a}}
.wrap{{max-width:900px;margin:0 auto;padding:24px}}
h1{{font-size:20px}}.teach{{background:#eef3f8;border:1px solid #cdd;border-radius:8px;padding:14px 18px;margin:16px 0}}
.teach p{{font-size:13.5px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:10px;padding:16px;margin:18px 0}}
.qn{{font:12px monospace;color:#666;margin-bottom:6px}}
.facet{{margin:5px 0;padding:7px 10px;background:#f4f2ee;border-radius:6px}}.facet b{{display:inline-block;min-width:90px;color:#3a6ea5}}
.nbrs{{margin:10px 0;border-top:1px dashed #ddd;padding-top:8px;font-size:13px}}
.nbr{{margin:4px 0}}.num{{font-weight:700;color:#b5651d;margin-right:4px}}.ab{{color:#666}}
details.ans{{margin-top:10px;border-top:2px solid #ddd;padding-top:8px}}details.ans summary{{cursor:pointer;font-weight:600;color:#3a6ea5}}
.fa{{font:13px monospace}}.hint{{font-size:12px;color:#666}}
@media(prefers-color-scheme:dark){{body{{background:#1c1b19;color:#eee}}.card{{background:#26251f;border-color:#444}}.facet{{background:#333127}}.teach{{background:#232a30;border-color:#456}}}}
</style></head><body><div class=wrap>
<h1>Annotation Practice (do before the real packet)</h1>
<p class=hint>Work each case, decide the three-state, then reveal the reference answer and compare. Discuss disagreements as a group before starting the real (no-answer) packet — after that, annotate independently.</p>
{guide}{rows}</div></body></html>'''


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/annotation')
    build(ap.parse_args().out)
