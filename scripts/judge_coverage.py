#!/usr/bin/env python3
"""Facet-coverage judgment pipeline (scaffold, v0.1).

对每条 contribution:检索时间截断近邻 → 构造事实性覆盖判定 prompt → LLM 判定 → 三态归类。
设计约束(docs/_methodology-notes.md 的铁律):
- LLM 只做逐 facet 的覆盖事实判断,必须引用近邻原文,禁整体 novelty 打分;
- 盲化:prompt 不含论文来源/系统名/人机身份;
- 判定模型与版本固定,全部输入输出落盘可复现。

用法:
  python3 judge_coverage.py --pid FA0007            # 单篇(调试)
  python3 judge_coverage.py --all                   # 全量
依赖环境变量(.env):OPENALEX_API_KEY, S2_API_KEY, ANTHROPIC_API_KEY(缺 S2/ANTHROPIC 时进入 dry-run,只产出检索结果与 prompt,不判定)。
"""
import argparse, json, os, re, sys, time, urllib.request, urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT_DIR = os.path.join(ROOT, 'data', 'extractions')
OUT_DIR = os.path.join(ROOT, 'data', 'judgments')
FARS_DATA = os.path.join(ROOT, 'data', 'fars-a-reviews', 'data')
K_NEIGHBORS = 15  # DECISION 2 (锁定): k=15, 消融 10/20

def load_env():
    env = {}
    p = os.path.join(ROOT, '.env')
    if os.path.exists(p):
        for line in open(p):
            if '=' in line and not line.strip().startswith('#'):
                k, v = line.strip().split('=', 1)
                env[k] = v
    env.update({k: os.environ[k] for k in
                ('OPENALEX_API_KEY', 'S2_API_KEY', 'ANTHROPIC_API_KEY') if k in os.environ})
    return env

ENV = load_env()

def http_json(url, headers=None, retries=3, backoff=8):
    req = urllib.request.Request(url, headers={'User-Agent': 'novelty-audit/0.1 (mailto:tan1@my.hpu.edu)', **(headers or {})})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(backoff * (i + 1)); continue
            raise
    raise RuntimeError(f'429 after {retries} retries: {url}')

def cutoff_for(pid):
    """时间截断:该论文 submission_date 前一天。"""
    pr = os.path.join(FARS_DATA, pid, 'paperreview.json')
    if os.path.exists(pr):
        d = json.load(open(pr)).get('submission_date', '')
        m = re.match(r'(\d{4}-\d{2}-\d{2})', d or '')
        if m: return m.group(1)
    return '2026-02-18'  # FARS 部署窗口起点前;非 FARS 语料在各自 loader 里覆盖

# ---------------- 检索层 ----------------

def search_s2(query, cutoff, limit=10):
    if 'S2_API_KEY' not in ENV:
        return None  # dry-run
    params = urllib.parse.urlencode({
        'query': query, 'limit': limit,
        'fields': 'title,abstract,year,publicationDate,externalIds',
        'publicationDateOrYear': f':{cutoff}'})
    return http_json(f'https://api.semanticscholar.org/graph/v1/paper/search?{params}',
                     headers={'x-api-key': ENV['S2_API_KEY']}).get('data', [])

def search_openalex(query, cutoff, limit=10):
    params = {'search': query, 'per-page': limit,
              'filter': f'to_publication_date:{cutoff}',
              'select': 'title,publication_date,abstract_inverted_index,doi'}
    if 'OPENALEX_API_KEY' in ENV:
        params['api_key'] = ENV['OPENALEX_API_KEY']
    res = http_json(f'https://api.openalex.org/works?{urllib.parse.urlencode(params)}')
    out = []
    for w in res.get('results', []):
        inv = w.get('abstract_inverted_index')
        ab = ''
        if inv:
            pos = {i: t for t, idxs in inv.items() for i in idxs}
            ab = ' '.join(pos[i] for i in sorted(pos))
        out.append({'title': w.get('title'), 'publicationDate': w.get('publication_date'),
                    'abstract': ab, 'doi': w.get('doi')})
    return out

def build_queries(contrib):
    """每条 contribution 生成多路查询。跨域规则:mechanism 查询绝不带 domain 词。"""
    return [
        ('purpose+domain', f"{contrib['purpose']} {contrib['domain']}"),
        ('mechanism-crossdomain', contrib['mechanism']),          # 关键:不限域
        ('purpose+mechanism', f"{contrib['purpose']} {contrib['mechanism']}"),
    ]

def retrieve_neighbors(contrib, cutoff, k=K_NEIGHBORS):
    """三路查询 × 双通道,合并去重,截取 top-k。TODO(S2 key 后): snippet search + recommendations(seed=论文自引)。"""
    seen, out = set(), []
    for tag, q in build_queries(contrib):
        for chan, fn in (('s2', search_s2), ('openalex', search_openalex)):
            try:
                hits = fn(q, cutoff) or []
            except Exception as e:
                print(f'    [{chan}:{tag}] {e}', file=sys.stderr); hits = []
            for h in hits:
                key = (h.get('title') or '').lower().strip()
                if key and key not in seen:
                    seen.add(key)
                    out.append({**h, 'via': f'{chan}:{tag}'})
            time.sleep(1.0)
    return out[:k]

# ---------------- 判定层 ----------------

JUDGE_PROMPT = """You are performing a factual coverage check for a bibliometric audit. You will see one research contribution, decomposed into four facets, and {n} prior papers (title + abstract), all published before the contribution was written.

For EACH facet (purpose, mechanism, evaluation, domain), determine whether any of the prior papers contains a substantively equivalent element. "Substantively equivalent" means the same core idea at the level of method/attack/measurement primitive; different naming, notation, scale, or surface application does not break equivalence. Judge mechanisms at the primitive level (e.g., "diluting a statistical signal by adding benign content" is one primitive regardless of the domain it is applied to).

Rules:
- This is a factual lookup task, NOT a quality or novelty assessment. Do not rate how good or original the contribution is.
- For every facet you mark as covered, you MUST cite the paper number and quote the supporting phrase from its title/abstract.
- If no listed paper covers a facet, mark it NONE. Do not use knowledge of papers not listed here.

CONTRIBUTION:
- purpose: {purpose}
- mechanism: {mechanism}
- evaluation: {evaluation}
- domain: {domain}

PRIOR PAPERS:
{papers}

Answer in JSON only:
{{"purpose": {{"covered_by": <paper number or null>, "evidence": "<quote or null>"}},
  "mechanism": {{"covered_by": ..., "evidence": ...}},
  "evaluation": {{"covered_by": ..., "evidence": ...}},
  "domain": {{"covered_by": ..., "evidence": ...}},
  "single_paper_covers_all": <paper number or null>}}"""

def classify(facets):
    """三态归类(DECISION 1 锁定:三态 + incremental 子标签在人工层处理)。"""
    if facets.get('single_paper_covers_all'):
        return 'covered'
    vals = [facets[f].get('covered_by') for f in ('purpose', 'mechanism', 'evaluation', 'domain')]
    if all(v is not None for v in vals):
        return 'recombination'
    return 'facet-novel'

def judge_llm(contrib, neighbors):
    if 'ANTHROPIC_API_KEY' not in ENV:
        return None  # dry-run: prompt 落盘,判定留空
    papers = '\n'.join(f"[{i+1}] {n['title']} — {(n.get('abstract') or '')[:600]}"
                       for i, n in enumerate(neighbors))
    prompt = JUDGE_PROMPT.format(n=len(neighbors), papers=papers, **{
        f: contrib[f] for f in ('purpose', 'mechanism', 'evaluation', 'domain')})
    body = json.dumps({'model': 'claude-sonnet-5', 'max_tokens': 1024,
                       'messages': [{'role': 'user', 'content': prompt}]}).encode()
    req = urllib.request.Request('https://api.anthropic.com/v1/messages', data=body, headers={
        'x-api-key': ENV['ANTHROPIC_API_KEY'], 'anthropic-version': '2023-06-01',
        'content-type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        text = json.load(r)['content'][0]['text']
    m = re.search(r'\{.*\}', text, re.S)
    return json.loads(m.group(0)) if m else {'raw': text}

def process(pid):
    src = os.path.join(EXTRACT_DIR, f'{pid}.json')
    if not os.path.exists(src):
        print(f'{pid}: no extraction'); return
    d = json.load(open(src))
    cutoff = cutoff_for(pid)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {'paper_id': pid, 'cutoff': cutoff, 'k': K_NEIGHBORS,
           'judge_model': 'claude-sonnet-5' if 'ANTHROPIC_API_KEY' in ENV else 'DRY-RUN',
           'contributions': []}
    for c in d['contributions']:
        print(f'  {pid}/{c["id"]} retrieving...', flush=True)
        nbrs = retrieve_neighbors(c, cutoff)
        facets = judge_llm(c, nbrs)
        out['contributions'].append({
            'id': c['id'], 'neighbors': nbrs, 'facet_coverage': facets,
            'state': classify(facets) if facets and 'raw' not in facets else None})
    json.dump(out, open(os.path.join(OUT_DIR, f'{pid}.json'), 'w'), ensure_ascii=False, indent=1)
    states = [c['state'] for c in out['contributions']]
    print(f'{pid}: {states}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid'); ap.add_argument('--all', action='store_true')
    a = ap.parse_args()
    pids = [a.pid] if a.pid else sorted(f[:-5] for f in os.listdir(EXTRACT_DIR) if f.endswith('.json')) if a.all else []
    if not pids: ap.print_help(); sys.exit(1)
    for pid in pids: process(pid)
