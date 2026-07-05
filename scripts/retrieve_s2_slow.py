#!/usr/bin/env python3
"""Slow-lane retrieval: keyless S2 main channel + OpenAlex second channel + bge-m3 rerank.

背景(2026-07-05 通宵档 7):S2 key 还在申请,Tao 拍板"开慢速跑着"。
S2 官方 API 无 key 可用,但走全球共享限流池,429 极常见。本脚本按"慢车道"设计:
- 每次 S2 请求间隔 >= S2_MIN_INTERVAL 秒(默认 8s),429 时指数退避(30s 起,上限 10min),永不放弃单条查询直到 MAX_429_STREAK;
- 断点续跑:每个贡献的候选落盘 data/neighbors_s2/<pid>.json,已完成的贡献跳过;
- 查询词由 KEYWORD_MODEL 生成并缓存(data/neighbors_s2/_queries.json),重跑不重复花 LLM;
- OpenAlex 通道照旧(它不限流),S2 拿不到时该贡献仍有 OpenAlex 候选,S2 拿到后合并去重、bge-m3 重排;
- 输出仅是"近邻候选缓存",不做任何判定(判定层等成本数字 + Tao 拍板)。

用法:
  python3 scripts/retrieve_s2_slow.py                # 全量 166 篇,慢速跑
  python3 scripts/retrieve_s2_slow.py --pid FA0007   # 单篇(验收用)
  python3 scripts/retrieve_s2_slow.py --status       # 看进度
"""
import argparse, json, math, os, random, re, sys, time, urllib.parse, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT_DIR = os.path.join(ROOT, 'data', 'extractions')
OUT_DIR = os.path.join(ROOT, 'data', 'neighbors_s2')
FARS_DATA = os.path.join(ROOT, 'data', 'fars-a-reviews', 'data')
QUERY_CACHE = os.path.join(OUT_DIR, '_queries.json')

def load_env():
    e = {}
    for l in open(os.path.join(ROOT, '.env')):
        if '=' in l and not l.strip().startswith('#'):
            k, v = l.split('=', 1); e[k.strip()] = v.strip().strip('"').strip("'")
    return e
ENV = load_env()

# 有 key:官方限速 1 req/s,留余量 1.2s;无 key:共享池,8s + 长退避
HAS_KEY = bool(ENV.get('S2_API_KEY'))
S2_MIN_INTERVAL = float(os.environ.get('S2_MIN_INTERVAL', 1.2 if HAS_KEY else 8))
MAX_429_STREAK = 40          # 连续 429 这么多次才放弃当前查询
BACKOFF_BASE, BACKOFF_CAP = (5, 60) if HAS_KEY else (30, 600)

def http(url, headers=None, data=None, retries=3, timeout=120):
    body = json.dumps(data).encode() if data is not None else None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, data=body,
                headers={'User-Agent': 'novelty-audit/0.3 (mailto:tan1@my.hpu.edu)', **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            if i == retries - 1: raise
            time.sleep(3)

def chat(model, prompt, max_tokens=300):
    """claude 走 uniapi 原生 /claude/v1/messages。注意:claude-sonnet-5 模型本身已弃用
    temperature(原生形式实测 400 "`temperature` is deprecated"),不是代理限制,故不传;
    其余模型走 OpenAI 兼容 /chat/completions,temperature=0(可复现)。"""
    if model.startswith('claude'):
        base = re.sub(r'/v1/?$', '', ENV['LLM_API_BASE'])
        r = http(f"{base}/claude/v1/messages",
                 {'x-api-key': ENV['LLM_API_KEY'], 'Content-Type': 'application/json',
                  'anthropic-version': '2023-06-01'},
                 {'model': model, 'max_tokens': max_tokens,
                  'messages': [{'role': 'user', 'content': prompt}]})
        return ''.join(b.get('text', '') for b in r.get('content', []))
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

# ---- query generation (cached) ----
# v0.4:两段式查询生成,都走 QUERY_MODEL(默认 claude-sonnet-5)。
# 教训(FA0007 验收):①facet 是中文时弱模型生成中文查询,S2/OpenAlex 全是垃圾 → 强制英文;
# ②常规关键词查询猜不中 seminal 论文的历史术语("good word attack"),
#   但让模型显式枚举"这个原语在早期文献里叫什么"能命中(claude 能,gpt-4o/mini 不能)。
QUERY_MODEL = os.environ.get('QUERY_MODEL', 'claude-sonnet-5')

KW_PROMPT = """Generate search queries to find PRIOR published work that might contain the same core idea as this research contribution. ENGLISH ONLY (translate non-English facets first). Output one query per line, no numbering, no headers, 3-6 words each, up to 8 lines total:
- 4 keyword queries: cover purpose, mechanism (as an abstract primitive WITHOUT the application domain, so cross-domain prior art surfaces), evaluation setup, and the specific technique name if any.
- up to 4 historical names: the same underlying primitive likely existed in OLDER or ADJACENT literatures (spam filtering, adversarial ML, IR, databases, statistics, security...) under established term-of-art NAMES. List those names; prefer real terms-of-art over descriptive paraphrases.

purpose: {purpose}
mechanism: {mechanism}
evaluation: {evaluation}
domain: {domain}"""

def load_query_cache():
    return json.load(open(QUERY_CACHE)) if os.path.exists(QUERY_CACHE) else {}

def save_query_cache(qc):
    json.dump(qc, open(QUERY_CACHE, 'w'), ensure_ascii=False, indent=1)

def _parse_lines(txt, n):
    return [l.strip(' -*').strip() for l in txt.splitlines() if l.strip()][:n]

def get_queries(pid, c, qc):
    # max_tokens 要够大:claude 推理模型的思考 tokens 计入 max_tokens,300 会把正文挤没(FA0007/C3 教训)
    key = f'{pid}/{c["id"]}/v4'
    if key not in qc:
        fmt = {k: c[k] for k in ('purpose','mechanism','evaluation','domain')}
        txt = chat(QUERY_MODEL, KW_PROMPT.format(**fmt), 2000)
        seen, qs = set(), []
        for q in _parse_lines(txt, 8):
            if q.lower() not in seen:
                seen.add(q.lower()); qs.append(q)
        if not qs:  # 回退弱模型;还空就不缓存,留给下轮重试
            qs = _parse_lines(chat(ENV['KEYWORD_MODEL'], KW_PROMPT.format(**fmt), 800), 8)
            if not qs: return []
        qc[key] = qs
        save_query_cache(qc)
    return qc[key]

# ---- S2 keyless slow lane ----
_last_s2 = [0.0]
def s2_search(query, cutoff, limit=20):
    """Returns (hits, ok). ok=False 表示放弃(连续 429 超限),调用方记账后继续。"""
    p = urllib.parse.urlencode({
        'query': query, 'limit': limit,
        'publicationDateOrYear': f':{cutoff}',
        'fields': 'title,abstract,publicationDate,year,externalIds,citationCount'})
    url = f'https://api.semanticscholar.org/graph/v1/paper/search?{p}'
    streak = 0
    while streak < MAX_429_STREAK:
        wait = _last_s2[0] + S2_MIN_INTERVAL - time.time()
        if wait > 0: time.sleep(wait)
        _last_s2[0] = time.time()
        try:
            hdrs = {'User-Agent': 'novelty-audit/0.3 (mailto:tan1@my.hpu.edu)'}
            if HAS_KEY: hdrs['x-api-key'] = ENV['S2_API_KEY']
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=60) as r:
                res = json.load(r)
            out = []
            for w in res.get('data') or []:
                if w.get('title'):
                    out.append({'title': w['title'],
                                'date': w.get('publicationDate') or (str(w['year']) if w.get('year') else None),
                                'abstract': (w.get('abstract') or '')[:800],
                                'doi': (w.get('externalIds') or {}).get('DOI'),
                                'citations': w.get('citationCount'),
                                'src': 's2'})
            return out, True
        except urllib.error.HTTPError as e:
            if e.code == 429:
                streak += 1
                delay = min(BACKOFF_CAP, BACKOFF_BASE * (1.6 ** min(streak, 7))) * (0.7 + 0.6 * random.random())
                print(f'    [s2 429 x{streak}, sleep {delay:.0f}s]', flush=True)
                time.sleep(delay)
                continue
            print(f'    [s2 http {e.code}, skip query]', flush=True)
            return [], False
        except Exception as ex:
            streak += 1
            print(f'    [s2 err {type(ex).__name__}, retry]', flush=True)
            time.sleep(15)
    return [], False

# ---- OpenAlex channel (unchanged from v0.2) ----
def openalex(query, cutoff, limit=25):
    p = {'search': query, 'per-page': limit, 'filter': f'to_publication_date:{cutoff}',
         'select': 'title,publication_date,abstract_inverted_index,doi'}
    if ENV.get('OPENALEX_API_KEY'): p['api_key'] = ENV['OPENALEX_API_KEY']
    try:
        res = http(f'https://api.openalex.org/works?{urllib.parse.urlencode(p)}')
    except Exception:
        return []
    out = []
    for w in res.get('results', []):
        inv = w.get('abstract_inverted_index'); ab = ''
        if inv:
            pos = {i: t for t, idxs in inv.items() for i in idxs}
            ab = ' '.join(pos[i] for i in sorted(pos))
        if w.get('title'):
            out.append({'title': w['title'], 'date': w.get('publication_date'),
                        'abstract': ab[:800], 'doi': w.get('doi'), 'src': 'openalex'})
    return out

# ---- per-contribution retrieval ----
def retrieve_one(pid, c, cutoff, qc, k):
    cand, seen = [], set()
    s2_ok_all = True
    queries = get_queries(pid, c, qc)
    if not queries:
        return {'id': c['id'], 's2_complete': False, 'neighbors': []}
    for q in queries:
        s2_hits, ok = s2_search(q, cutoff)
        s2_ok_all = s2_ok_all and ok
        # 每条查询保送两篇进最终近邻:相关性 top-1 + 前 5 名里引用数最高者。
        # 防止 seminal 论文(老、高引、短/无摘要,如 Lowd&Meek 2005 在"good word attack"下排第 3)
        # 被 embedding 重排或 top-1 截断切掉。
        for rank, h in enumerate(s2_hits):
            h['pinned'] = (rank == 0)
        top5 = s2_hits[:5]
        if top5:
            most_cited = max(top5, key=lambda x: x.get('citations') or 0)
            if (most_cited.get('citations') or 0) > 0: most_cited['pinned'] = True
        for h in s2_hits + openalex(q, cutoff):
            key = h['title'].lower().strip()
            if key not in seen:
                seen.add(key); cand.append(h)
            elif h.get('pinned'):
                for e in cand:
                    if e['title'].lower().strip() == key: e['pinned'] = True; break
        time.sleep(0.5)
    if not cand:
        return {'id': c['id'], 's2_complete': s2_ok_all, 'neighbors': []}
    ctext = f"{c['purpose']} {c['mechanism']} {c['evaluation']}"
    vecs = embed([ctext] + [f"{h['title']}. {h['abstract']}" for h in cand])
    cv, hv = vecs[0], vecs[1:]
    for h, v in zip(cand, hv): h['sim'] = round(cos(cv, v), 4)
    cand.sort(key=lambda x: x['sim'], reverse=True)
    pinned = [h for h in cand if h.get('pinned')]
    rest = [h for h in cand if not h.get('pinned')]
    nbrs = (pinned + rest)[:max(k, len(pinned))]
    nbrs.sort(key=lambda x: x['sim'], reverse=True)
    return {'id': c['id'], 's2_complete': s2_ok_all,
            'n_candidates': len(cand), 'n_pinned': len(pinned), 'neighbors': nbrs}

def out_path(pid): return os.path.join(OUT_DIR, f'{pid}.json')

def done_cids(pid):
    """已完成且 S2 通道完整的贡献 id 集合(S2 不完整的会重跑)。"""
    if not os.path.exists(out_path(pid)): return set(), None
    d = json.load(open(out_path(pid)))
    return {c['id'] for c in d['contributions'] if c.get('s2_complete')}, d

def process(pid, k):
    d = json.load(open(os.path.join(EXTRACT_DIR, f'{pid}.json')))
    cutoff = cutoff_for(pid)
    qc = load_query_cache()
    done, existing = done_cids(pid)
    todo = [c for c in d['contributions'] if c['id'] not in done]
    if not todo:
        return 0
    out = existing or {'paper_id': pid, 'cutoff': cutoff, 'k': k,
                       'retrieval': 's2-keyless+openalex+bge-m3-rerank', 'contributions': []}
    kept = [c for c in out['contributions'] if c['id'] in done]
    for c in todo:
        print(f'  {pid}/{c["id"]} ...', flush=True)
        r = retrieve_one(pid, c, cutoff, qc, k)
        kept.append(r)
        out['contributions'] = kept
        json.dump(out, open(out_path(pid), 'w'), ensure_ascii=False, indent=1)
        n_s2 = sum(1 for h in r['neighbors'] if h['src'] == 's2')
        print(f'    -> {len(r["neighbors"])} nbrs (s2 {n_s2}), s2_complete={r["s2_complete"]}', flush=True)
    return len(todo)

def status():
    pids = sorted(f[:-5] for f in os.listdir(EXTRACT_DIR) if f.endswith('.json'))
    total_c = done_c = 0
    for pid in pids:
        d = json.load(open(os.path.join(EXTRACT_DIR, f'{pid}.json')))
        n = len(d['contributions']); total_c += n
        done, _ = done_cids(pid)
        done_c += len(done)
    print(f'papers: {sum(1 for p in pids if done_cids(p)[0])}/{len(pids)} touched; contributions s2-complete: {done_c}/{total_c}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid'); ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--status', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.status:
        status(); sys.exit(0)
    pids = [a.pid] if a.pid else sorted(f[:-5] for f in os.listdir(EXTRACT_DIR) if f.endswith('.json'))
    t0 = time.time()
    for i, pid in enumerate(pids):
        n = process(pid, a.k)
        if n:
            print(f'[{i+1}/{len(pids)}] {pid} +{n} contribs, elapsed {int(time.time()-t0)}s', flush=True)
    print('ALL DONE', flush=True)
