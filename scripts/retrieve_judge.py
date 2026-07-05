#!/usr/bin/env python3
"""Retrieval + facet-coverage judgment, v0.2 (real LLM + bge-m3 rerank).

改进(相对 judge_coverage.py v0.1 dry-run):
- LLM 生成检索关键词(不再把长句 facet 塞进关键词检索)——修 v0.1 的召回噪声问题;
- OpenAlex 拉候选 → bge-m3 embedding 相似度重排 → top-k(SPECTER-2 的替代,且与 UIUC 指标同 embedding);
- gpt-4o 事实性覆盖判定(盲化、逐 facet、强制引证、禁裸打分)。
仍待 S2 key 补主检索通道 + 第二家族裁判(多裁判一致性)。

用法: python3 retrieve_judge.py --pid FA0007 [--k 15]
"""
import argparse, json, os, re, sys, time, math, urllib.request, urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT_DIR = os.path.join(ROOT, 'data', 'extractions')
OUT_DIR = os.path.join(ROOT, 'data', 'judgments')
FARS_DATA = os.path.join(ROOT, 'data', 'fars-a-reviews', 'data')

def load_env():
    e = {}
    for l in open(os.path.join(ROOT, '.env')):
        if '=' in l and not l.strip().startswith('#'):
            k, v = l.split('=', 1); e[k.strip()] = v.strip().strip('"').strip("'")
    return e
ENV = load_env()

def http(url, headers=None, data=None, retries=4, backoff=6):
    body = json.dumps(data).encode() if data is not None else None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, data=body,
                headers={'User-Agent': 'novelty-audit/0.2 (mailto:tan1@my.hpu.edu)', **(headers or {})})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429: time.sleep(backoff * (i + 1)); continue
            if i == retries - 1: raise
            time.sleep(2)
        except Exception:
            if i == retries - 1: raise
            time.sleep(2)
    raise RuntimeError(f'failed: {url}')

def chat(model, prompt, max_tokens=800):
    r = http(f"{ENV['LLM_API_BASE']}/chat/completions",
             {'Authorization': f"Bearer {ENV['LLM_API_KEY']}", 'Content-Type': 'application/json'},
             {'model': model, 'max_tokens': max_tokens, 'temperature': 0,
              'messages': [{'role': 'user', 'content': prompt}]})
    return r['choices'][0]['message']['content']

def embed(texts):
    r = http(f"{ENV['EMBED_API_BASE']}/embeddings",
             {'Authorization': 'Bearer NO_NEED', 'Content-Type': 'application/json'},
             {'model': ENV['EMBED_MODEL'], 'input': texts})
    return [d['embedding'] for d in r['data']]

def cos(a, b):
    s = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return s/(na*nb) if na and nb else 0.0

def cutoff_for(pid):
    pr = os.path.join(FARS_DATA, pid, 'paperreview.json')
    if os.path.exists(pr):
        m = re.match(r'(\d{4}-\d{2}-\d{2})', json.load(open(pr)).get('submission_date', '') or '')
        if m: return m.group(1)
    return '2026-02-18'

# ---- retrieval ----
KW_PROMPT = """Generate search queries to find PRIOR published work that might contain the same core idea as this research contribution. Output 4 short keyword queries (3-6 words each), one per line, no numbering.
Rules: one query MUST target the mechanism as an abstract primitive WITHOUT the application domain (so cross-domain prior art surfaces). Cover purpose, mechanism (domain-free), evaluation setup, and the specific technique name if any.

purpose: {purpose}
mechanism: {mechanism}
evaluation: {evaluation}
domain: {domain}"""

def gen_queries(c):
    txt = chat(ENV['KEYWORD_MODEL'], KW_PROMPT.format(**{k: c[k] for k in ('purpose','mechanism','evaluation','domain')}), 200)
    qs = [l.strip(' -*').strip() for l in txt.splitlines() if l.strip()]
    return qs[:4]

def openalex(query, cutoff, limit=25):
    p = {'search': query, 'per-page': limit, 'filter': f'to_publication_date:{cutoff}',
         'select': 'title,publication_date,abstract_inverted_index,doi'}
    if ENV.get('OPENALEX_API_KEY'): p['api_key'] = ENV['OPENALEX_API_KEY']
    res = http(f'https://api.openalex.org/works?{urllib.parse.urlencode(p)}')
    out = []
    for w in res.get('results', []):
        inv = w.get('abstract_inverted_index'); ab = ''
        if inv:
            pos = {i: t for t, idxs in inv.items() for i in idxs}
            ab = ' '.join(pos[i] for i in sorted(pos))
        if w.get('title'):
            out.append({'title': w['title'], 'date': w.get('publication_date'),
                        'abstract': ab[:800], 'doi': w.get('doi')})
    return out

def retrieve(c, cutoff, k):
    cand, seen = [], set()
    for q in gen_queries(c):
        for h in openalex(q, cutoff):
            key = h['title'].lower().strip()
            if key not in seen:
                seen.add(key); cand.append(h)
        time.sleep(0.8)
    if not cand: return []
    # bge-m3 rerank against the contribution text
    ctext = f"{c['purpose']} {c['mechanism']} {c['evaluation']}"
    vecs = embed([ctext] + [f"{h['title']}. {h['abstract']}" for h in cand])
    cv, hv = vecs[0], vecs[1:]
    for h, v in zip(cand, hv): h['sim'] = round(cos(cv, v), 4)
    cand.sort(key=lambda x: x['sim'], reverse=True)
    return cand[:k]

# ---- judgment ----
JUDGE = """You are performing a factual coverage check for a bibliometric audit. Below is one research contribution decomposed into four facets, and {n} prior papers (published before it). For EACH facet decide whether any prior paper contains a substantively equivalent element.
"Substantively equivalent" = same core idea at the level of method/attack/measurement primitive; different naming, notation, scale, or surface application does NOT break equivalence. Judge mechanisms at the primitive level (e.g. "diluting a statistical signal by adding benign tokens" is one primitive regardless of application domain).
This is a FACTUAL LOOKUP, not a novelty/quality rating. For every facet you mark covered you MUST cite the paper number and quote the supporting phrase. If nothing listed covers a facet, use null. Use ONLY the listed papers.

CONTRIBUTION:
- purpose: {purpose}
- mechanism: {mechanism}
- evaluation: {evaluation}
- domain: {domain}

PRIOR PAPERS:
{papers}

Output JSON only:
{{"purpose":{{"covered_by":<n|null>,"evidence":"<quote|null>"}},"mechanism":{{"covered_by":<n|null>,"evidence":"..."}},"evaluation":{{"covered_by":<n|null>,"evidence":"..."}},"domain":{{"covered_by":<n|null>,"evidence":"..."}},"single_paper_covers_all":<n|null>}}"""

def judge(c, nbrs):
    papers = '\n'.join(f"[{i+1}] ({n.get('date')}) {n['title']} — {n['abstract'][:500]}" for i, n in enumerate(nbrs))
    txt = chat(ENV['JUDGE_MODEL'], JUDGE.format(n=len(nbrs), papers=papers,
               **{k: c[k] for k in ('purpose','mechanism','evaluation','domain')}), 900)
    m = re.search(r'\{.*\}', txt, re.S)
    return json.loads(m.group(0)) if m else {'raw': txt}

def classify(f):
    if not f or 'raw' in f: return None
    if f.get('single_paper_covers_all'): return 'covered'
    vals = [f[x].get('covered_by') for x in ('purpose','mechanism','evaluation','domain')]
    return 'recombination' if all(v is not None for v in vals) else 'facet-novel'

def process(pid, k):
    d = json.load(open(os.path.join(EXTRACT_DIR, f'{pid}.json')))
    cutoff = cutoff_for(pid)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {'paper_id': pid, 'cutoff': cutoff, 'k': k, 'judge_model': ENV['JUDGE_MODEL'],
           'embed_model': ENV['EMBED_MODEL'], 'retrieval': 'openalex+bge-m3-rerank', 'contributions': []}
    for c in d['contributions']:
        print(f'  {pid}/{c["id"]} ...', end='', flush=True)
        nbrs = retrieve(c, cutoff, k)
        fac = judge(c, nbrs) if nbrs else None
        st = classify(fac)
        out['contributions'].append({'id': c['id'], 'state': st,
            'facet_coverage': fac, 'neighbors': nbrs})
        print(f' {st} ({len(nbrs)} nbrs, top sim {nbrs[0]["sim"] if nbrs else "NA"})')
    json.dump(out, open(os.path.join(OUT_DIR, f'{pid}.json'), 'w'), ensure_ascii=False, indent=1)
    print(f'{pid}: {[c["state"] for c in out["contributions"]]}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', required=True); ap.add_argument('--k', type=int, default=15)
    a = ap.parse_args()
    process(a.pid, a.k)
