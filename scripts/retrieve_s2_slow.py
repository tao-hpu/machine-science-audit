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
  python3 scripts/retrieve_s2_slow.py                # 全量 FARS,慢速跑
  python3 scripts/retrieve_s2_slow.py --pid FA0007   # 单篇(验收用)
  python3 scripts/retrieve_s2_slow.py --status       # 看进度
  # v6 机器侧 per-facet 重选(查询缓存先从 v5 目录拷入新 out-dir):
  python3 scripts/retrieve_s2_slow.py --facet-alloc --out-dir data/neighbors_s2_v6
  # 人类基线(cutoff 用每篇 cdate,不是年份!):
  python3 scripts/retrieve_s2_slow.py --extract-dir data/extractions_human \
      --out-dir data/neighbors_human --facet-alloc --cutoff-source human
  # A4S(cutoff 用抽取里的 submission_date / submissions.json cdate):
  python3 scripts/retrieve_s2_slow.py --extract-dir data/extractions_a4s \
      --out-dir data/neighbors_a4s --facet-alloc --cutoff-source a4s
"""
import argparse, datetime, json, math, os, random, re, sys, time, urllib.parse, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FARS_DATA = os.path.join(ROOT, 'data', 'fars-a-reviews', 'data')
# 下面四个由 main() 按参数覆盖;模块级默认保持旧行为(FARS v5)
EXTRACT_DIR = os.path.join(ROOT, 'data', 'extractions')
OUT_DIR = os.path.join(ROOT, 'data', 'neighbors_s2')
QUERY_CACHE = os.path.join(OUT_DIR, '_queries.json')
CUTOFF_SOURCE = 'fars'

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

def cross_rerank(query, cand, bs=256):
    """bge-reranker cross-encoder:query 对每个候选 (title. abstract) 联合打相关性分,原地写 h['sim']。
    双塔 bge-m3 靠余弦被通用词面骗(把 CPU/STT-RAM 硬件缓存排在 LLM KV cache 前);cross-encoder
    联合读两段文本能做领域感知判别,是本地重排逼近 S2 API 相关性的关键(docs/_local-index-report.md)。"""
    model = ENV.get('RERANK_MODEL') or 'bge-reranker-v2-m3'
    docs = [f"{h['title']}. {h.get('abstract','')}"[:1500] for h in cand]
    for i in range(0, len(docs), bs):
        chunk = docs[i:i + bs]
        r = http(f"{ENV['RERANK_API_BASE']}/rerank",
                 {'Authorization': 'Bearer NO_NEED', 'Content-Type': 'application/json'},
                 {'model': model, 'query': query[:1200], 'documents': chunk,
                  'top_n': len(chunk), 'return_documents': False})
        for x in r['results']:
            cand[i + x['index']]['sim'] = round(x['relevance_score'], 4)
    for h in cand:
        h.setdefault('sim', 0.0)
    return cand

def cos(a, b):
    s = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return s/(na*nb) if na and nb else 0.0

_CDATE_MAPS = {}
def _cdate_map(path):
    """OpenReview 导出的 cdate 是毫秒时间戳 → UTC 日期字符串。"""
    if path not in _CDATE_MAPS:
        rows = json.load(open(path))
        _CDATE_MAPS[path] = {
            r['id']: datetime.datetime.fromtimestamp(r['cdate'] / 1000, datetime.timezone.utc).strftime('%Y-%m-%d')
            for r in rows if r.get('cdate')}
    return _CDATE_MAPS[path]

def cutoff_for(pid, ext=None):
    """三种 cutoff 口径,与机器侧同构(每篇自己的提交日,绝不用整年):
    fars  = fars-a-reviews paperreview.json 的 submission_date;
    human = human_iclr2025/matched_166.json 按 paper_id 查 cdate;
    a4s   = 抽取自带 submission_date,缺则回查 agents4science/submissions.json(forum→cdate)。
    human/a4s 查不到直接抛错——静默回退到晚 cutoff 会把投稿后文献算进先行工作,污染 RQ1。"""
    if CUTOFF_SOURCE == 'human':
        m = _cdate_map(os.path.join(ROOT, 'data', 'human_iclr2025', 'matched_166.json'))
        if pid not in m:
            raise KeyError(f'{pid}: not in matched_166.json, no cutoff')
        return m[pid]
    if CUTOFF_SOURCE == 'a4s':
        sd = (ext or {}).get('submission_date', '') or ''
        mm = re.match(r'(\d{4}-\d{2}-\d{2})', sd)
        if mm: return mm.group(1)
        m = _cdate_map(os.path.join(ROOT, 'data', 'agents4science', 'submissions.json'))
        forum = (ext or {}).get('forum')
        if forum in m: return m[forum]
        raise KeyError(f'{pid}: no submission_date and forum {forum!r} not in submissions.json')
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

CROSS-DOMAIN REACH — CRITICAL GUARDRAIL: only reach into another field when that field studies the SAME core research idea. Do NOT generate a query just because both share a generic mathematical tool, component, or buzzword (e.g. "adaptive", "closed-loop", "control", "exponential moving average / EMA", "attention", "gradient", "threshold", "monitoring"). A shared generic tool is NOT a shared idea.
Example of a GOOD cross-domain reach: "prepend benign tokens to evade a classifier" (LLM safety) -> "good word attack" (spam filtering) — same core idea (signal-dilution evasion).
Example of a BAD cross-domain reach: "EMA-thresholded adaptive control loop during fine-tuning" (LLM alignment) -> "EWMA control chart" (statistical process control) — only a shared math tool (moving average), the research ideas are unrelated. Do NOT emit such queries.
If unsure whether a field truly shares the idea, stay within the contribution's own research problem rather than emit a speculative cross-field term.

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
    key = f'{pid}/{c["id"]}/v5'  # v5: 加跨域误联想护栏(禁「仅共享通用工具」的查询)
    if key not in qc:
        # 只读回退旧版本缓存:v6 per-facet 重选的消融口径是「查询不变,只换分配」。
        # 现状:坏近邻重跑过的 159 条有 v5 查询,其余 390 条只有 v4——这正是当前
        # v5 近邻池各自实际用过的查询,直接复用,不重新烧 LLM、不污染对照。
        for old_ver in ('v4',):
            old_key = f'{pid}/{c["id"]}/{old_ver}'
            if old_key in qc:
                return qc[old_key]
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
_OA_KEY_DEAD = [False]
def openalex(query, cutoff, limit=25):
    p = {'search': query, 'per-page': limit, 'filter': f'to_publication_date:{cutoff}',
         'select': 'title,publication_date,abstract_inverted_index,doi',
         'mailto': 'tan1@my.hpu.edu'}
    # 2026-07-08 教训:premium key 配额耗尽/失效时 OpenAlex 对带 key 请求一律 429,
    # 而无 key polite pool(带 mailto)完全正常 —— key 撞 429 就永久降级为无 key。
    if ENV.get('OPENALEX_API_KEY') and not _OA_KEY_DEAD[0]:
        p['api_key'] = ENV['OPENALEX_API_KEY']
    try:
        res = http(f'https://api.openalex.org/works?{urllib.parse.urlencode(p)}', retries=2)
    except urllib.error.HTTPError as e:
        if e.code == 429 and 'api_key' in p:
            _OA_KEY_DEAD[0] = True
            print('    [openalex key 429 -> fallback keyless]', flush=True)
            p.pop('api_key')
            try:
                res = http(f'https://api.openalex.org/works?{urllib.parse.urlencode(p)}', retries=2)
            except Exception:
                return []
        else:
            return []
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

# ---- channel dispatch(参数硬切,无跨通道降级;Tao 2026-07-08 定)----
# CHANNEL: 主检索通道。api=S2 线上(1req/s+429 退避);local=本地 tantivy 快照索引。
# OA_CHANNEL: 第二通道。api=OpenAlex 线上;local=本地 OA works 索引(待交付);off=关闭。
CHANNEL = 'api'
OA_CHANNEL = 'api'
DUMP_POOL = False  # --dump-pool:候选池+全 facet 相似度落盘 OUT_DIR/_pools/,供 Block A 离线消融
# 重排口径:facet=bge-m3 per-facet 名额(默认);embed=bge-m3 全局余弦;cross=深召回+bge-reranker
# cross-encoder(领域感知,治双塔词面混淆)。cross 模式自动深挖 BM25 召回(LOCAL_RECALL)。
RERANK_MODE = os.environ.get('RERANK_MODE', 'facet')
PIN_DEPTH = int(os.environ.get('PIN_DEPTH', '1'))  # 每查询保送 top-N(默认 1);跨域 seminal 常排 #2-3,历史术语查询调 3 兜住
LOCAL_RECALL = int(os.environ.get('LOCAL_RECALL', '500'))   # cross 模式每查询每索引召回深度
CITO_EXPAND = os.environ.get('CITO_EXPAND') == '1'  # cito 服务端跨域查询扩展(方案 #1);默认关,gate 对照用 =1 开
SEMINAL_RESCUE = os.environ.get('SEMINAL_RESCUE', '1') == '1'  # 跨域 seminal 救援 pin(选择层修复);默认开,=0 可消融
SEMINAL_RESCUE_CAP = int(os.environ.get('SEMINAL_RESCUE_CAP', '1'))  # 每条术语查询最多救援保送几篇(防灌水)

# 词面覆盖率用的最小停用词(只去真·虚词,attack/spam 这种内容词必须留)
_STOP = {'a', 'an', 'the', 'of', 'on', 'in', 'to', 'for', 'and', 'or', 'with', 'without',
         'via', 'using', 'into', 'from', 'by', 'is', 'are', 'be', 'as', 'at', 'this',
         'that', 'these', 'those', 'it', 'its', 'we', 'our'}

def _content_tokens(s):
    """标题/查询 → 内容词集合(去虚词、去 <3 字、轻量去复数尾 s),用于词面覆盖率匹配。"""
    toks = re.findall(r'[a-z0-9]+', (s or '').lower())
    return {t[:-1] if len(t) > 3 and t.endswith('s') else t
            for t in toks if len(t) > 2 and t not in _STOP}

def primary_search(query, cutoff):
    if CHANNEL == 'local':
        sys.path.insert(0, os.path.join(ROOT, 'scripts'))
        import local_search as LS
        if RERANK_MODE == 'cross':
            LS.POOL = max(LS.POOL, LOCAL_RECALL * 2)         # BM25 先取更深,cutoff 过滤后仍够
            return LS.search(query, cutoff, limit=LOCAL_RECALL), True
        return LS.search(query, cutoff, limit=20), True
    if CHANNEL == 'cito':
        # cito 私有混合检索(S2 148M+SPECTER2,自托管无限流)。cutoff→published_before(服务端过滤)。
        # 摘要默认截 800 与 s2_search 基线对齐;top-6 在题率只看标题,截断不影响 gate 指标。
        sys.path.insert(0, os.path.join(ROOT, 'scripts'))
        import cito_search as CS
        depth = LOCAL_RECALL if RERANK_MODE == 'cross' else 20
        return CS.search(query, cutoff, limit=depth, expand=CITO_EXPAND), True
    return s2_search(query, cutoff)

def oa_search(query, cutoff):
    if OA_CHANNEL == 'off':
        return []
    if OA_CHANNEL == 'local':
        sys.path.insert(0, os.path.join(ROOT, 'scripts'))
        import oa_local_search as OLS  # 交付物(docs/_local-index-task.md 追加节);未就绪即 ImportError,不静默回 API
        if RERANK_MODE == 'cross':
            OLS.POOL = max(OLS.POOL, LOCAL_RECALL * 2)
            return OLS.search(query, cutoff, limit=LOCAL_RECALL)
        return OLS.search(query, cutoff, limit=25)
    return openalex(query, cutoff)

# ---- per-contribution retrieval ----
def retrieve_one(pid, c, cutoff, qc, k):
    cand, seen = [], set()
    s2_ok_all = True
    queries = get_queries(pid, c, qc)
    if not queries:
        return {'id': c['id'], 's2_complete': False, 'neighbors': []}
    for q in queries:
        s2_hits, ok = primary_search(q, cutoff)
        s2_ok_all = s2_ok_all and ok
        # 每条查询保送两篇进最终近邻:相关性 top-1 + 前 5 名里引用数最高者。
        # 防止 seminal 论文(老、高引、短/无摘要,如 Lowd&Meek 2005 在"good word attack"下排第 3)
        # 被 embedding 重排或 top-1 截断切掉。
        for rank, h in enumerate(s2_hits):
            h['pinned'] = (rank < PIN_DEPTH)
        top5 = s2_hits[:5]
        if top5:
            most_cited = max(top5, key=lambda x: x.get('citations') or 0)
            if (most_cited.get('citations') or 0) > 0: most_cited['pinned'] = True
        for h in s2_hits + oa_search(q, cutoff):
            key = h['title'].lower().strip()
            if key not in seen:
                seen.add(key); cand.append(h)
            elif h.get('pinned'):
                for e in cand:
                    if e['title'].lower().strip() == key: e['pinned'] = True; break
        time.sleep(0.5 if 'api' in (CHANNEL, OA_CHANNEL) else 0.02)
    if not cand:
        return {'id': c['id'], 's2_complete': s2_ok_all, 'neighbors': []}
    # 跨域 seminal 救援 pin(选择层修复):bge-m3 facet 重排会低估跨域先例(领域/语义都远,如
    # LLM-safety 贡献 vs 2006 垃圾邮件论文),而这类先例往往只被「字面点名」的手工术语查询召回
    # (如 "Good word attack on spam filters")、又非任一查询 top-1/最高引 → pin 与 facet 双漏网。
    # 修法:某查询的内容词若被池中某标题高覆盖(≥0.7),就把那篇按词面直接保送——不靠 embedding
    # rank、不靠引用、不管来自哪个通道。阈值(query 内容词 ≥3 且覆盖 ≥0.7)保证泛概念查询不误触。
    if SEMINAL_RESCUE:
        cand_tok = [(_content_tokens(h['title']), h) for h in cand]
        for q in queries:
            qtok = _content_tokens(q)
            if len(qtok) < 4:      # 具体术语名(≥4 内容词)才救援;泛领域名(2-3 词,如
                continue           # "error correction codes"/"anomaly detection")会误命中该领域随机论文,跳过
            # 覆盖率达标者按 (cov 高→低, 标题短→长) 排序:同 cov 时优先短标题
            # ——原始 seminal 用规范简称(如 "Good Word Attacks on Statistical Spam Filters"),
            # 衍生/防御论文加前缀更长("Combating…"/"A Multiple Instance Learning Strategy for…")。
            scored = sorted(
                ((len(qtok & ttok) / len(qtok), len(h['title']), h) for ttok, h in cand_tok if ttok),
                key=lambda x: (-x[0], x[1]))
            for cov, _, h in scored[:SEMINAL_RESCUE_CAP]:
                if cov < 0.7:
                    break
                h['pinned'] = True
                h.setdefault('pin_reason', []).append('term_match')
    htexts = [f"{h['title']}. {h['abstract']}" for h in cand]
    if FACET_ALLOC and RERANK_MODE != 'cross':
        nbrs, n_pinned = select_per_facet(c, cand, htexts, k)
        if DUMP_POOL:
            import gzip
            pdir = os.path.join(OUT_DIR, '_pools')
            os.makedirs(pdir, exist_ok=True)
            with gzip.open(os.path.join(pdir, f'{pid}_{c["id"]}.json.gz'), 'wt') as pf:
                json.dump({'pid': pid, 'cid': c['id'], 'cutoff': cutoff, 'k': k,
                           'candidates': cand}, pf, ensure_ascii=False)
    else:
        if RERANK_MODE == 'cross':
            # 领域锚定查询(domain 圈领域 + purpose/mechanism 定具体 idea):实测把最难的
            # FA0002/C1 从 2→4 在题,并清掉硬件缓存误命中。
            q_ce = f"{c.get('domain','')}. {c.get('purpose','')} {c.get('mechanism','')}".strip()
            cross_rerank(q_ce, cand)                              # 原地写 h['sim']
        else:
            ctext = f"{c['purpose']} {c['mechanism']} {c['evaluation']}"
            vecs = embed([ctext] + htexts)
            cv, hv = vecs[0], vecs[1:]
            for h, v in zip(cand, hv): h['sim'] = round(cos(cv, v), 4)
        cand.sort(key=lambda x: x['sim'], reverse=True)
        pinned = [h for h in cand if h.get('pinned')]
        rest = [h for h in cand if not h.get('pinned')]
        nbrs = (pinned + rest)[:max(k, len(pinned))]
        nbrs.sort(key=lambda x: x['sim'], reverse=True)
        n_pinned = len(pinned)
        if DUMP_POOL:
            import gzip
            pdir = os.path.join(OUT_DIR, '_pools')
            os.makedirs(pdir, exist_ok=True)
            with gzip.open(os.path.join(pdir, f'{pid}_{c["id"]}.json.gz'), 'wt') as pf:
                json.dump({'pid': pid, 'cid': c['id'], 'cutoff': cutoff, 'k': k,
                           'candidates': cand}, pf, ensure_ascii=False)
    return {'id': c['id'], 's2_complete': s2_ok_all,
            'n_candidates': len(cand), 'n_pinned': n_pinned, 'neighbors': nbrs}


FACET_ALLOC = os.environ.get('FACET_ALLOC') == '1'  # main() 里 --facet-alloc 可覆盖
FACET_KEYS = ('purpose', 'mechanism', 'evaluation', 'domain')

def select_per_facet(c, cand, htexts, k):
    """按 facet 分配名额:每个 facet 用自己的文本单独算相似度,各取 top-⌈k/n⌉,并集 + pinned。
    防止最突出的单一 facet 垄断全局 top-k、饿死其他 facet 的先行工作。
    每个近邻打 facet_for 标签,说明它是为哪个 facet 保送进来的。
    副作用(供 --dump-pool 消融复放):给每个候选原地写 facet_sims(全部 facet 的
    相似度)与 sim_global(v5 口径 ctext 相似度)——Block A 的 k 扫描/alloc on-off/
    pinning on-off 全部可离线纯算术重放,零检索零 embedding。"""
    facets = [(f, c[f]) for f in FACET_KEYS if c.get(f) and c[f].strip()]
    ctext = f"{c.get('purpose','')} {c.get('mechanism','')} {c.get('evaluation','')}"
    vecs = embed([t for _, t in facets] + [ctext] + htexts)
    fvecs = vecs[:len(facets)]; cvec = vecs[len(facets)]; hv = vecs[len(facets) + 1:]
    for h, v in zip(cand, hv):
        h['facet_sims'] = {fname: round(cos(fv, v), 4) for (fname, _), fv in zip(facets, fvecs)}
        h['sim_global'] = round(cos(cvec, v), 4)
    per = max(1, -(-k // len(facets)))  # ceil(k/n)
    chosen = {}
    for fname, _ in facets:
        scored = sorted(((h['facet_sims'][fname], h) for h in cand),
                        key=lambda x: x[0], reverse=True)
        for sim, h in scored[:per]:
            key = h['title'].lower().strip()
            if key not in chosen:
                h = dict(h); h['sim'] = sim; h['facet_for'] = [fname]; chosen[key] = h
            else:
                chosen[key]['facet_for'].append(fname)
                chosen[key]['sim'] = max(chosen[key]['sim'], sim)
    # pinned(seminal 保送)一律纳入
    for h in cand:
        key = h['title'].lower().strip()
        if h.get('pinned') and key not in chosen:
            best = max(h['facet_sims'].values(), default=0)
            h = dict(h); h['sim'] = best; h['facet_for'] = ['pinned']; chosen[key] = h
    nbrs = sorted(chosen.values(), key=lambda x: x['sim'], reverse=True)
    n_pinned = sum(1 for h in nbrs if 'pinned' in h.get('facet_for', []))
    return nbrs, n_pinned

def out_path(pid): return os.path.join(OUT_DIR, f'{pid}.json')

def done_cids(pid):
    """已完成且 S2 通道完整的贡献 id 集合(S2 不完整的会重跑)。"""
    if not os.path.exists(out_path(pid)): return set(), None
    d = json.load(open(out_path(pid)))
    return {c['id'] for c in d['contributions'] if c.get('s2_complete')}, d

def process(pid, k):
    d = json.load(open(os.path.join(EXTRACT_DIR, f'{pid}.json')))
    cutoff = cutoff_for(pid, d)
    qc = load_query_cache()
    done, existing = done_cids(pid)
    todo = [c for c in d['contributions'] if c['id'] not in done]
    if not todo:
        return 0
    out = existing or {'paper_id': pid, 'cutoff': cutoff, 'k': k,
                       'cutoff_source': CUTOFF_SOURCE, 'facet_alloc': FACET_ALLOC,
                       'retrieval': f's2[{CHANNEL}]+oa[{OA_CHANNEL}]+rerank[{RERANK_MODE}]', 'contributions': []}
    kept = [c for c in out['contributions'] if c['id'] in done]
    for c in todo:
        print(f'  {pid}/{c["id"]} ...', flush=True)
        try:
            r = retrieve_one(pid, c, cutoff, qc, k)
        except Exception as ex:
            # 网络抖动等单点失败不杀全量:不落盘该贡献(留给下轮续跑重试),歇口气继续
            print(f'    !! {type(ex).__name__}: {str(ex)[:120]} — skip, retry next resume', flush=True)
            time.sleep(30)
            continue
        kept.append(r)
        out['contributions'] = kept
        json.dump(out, open(out_path(pid), 'w'), ensure_ascii=False, indent=1)
        n_s2 = sum(1 for h in r['neighbors'] if h['src'] in ('s2', 's2local'))  # local 通道 src=s2local
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
    ap.add_argument('--extract-dir', default=EXTRACT_DIR, help='抽取目录(默认 FARS data/extractions)')
    ap.add_argument('--out-dir', default=OUT_DIR, help='近邻输出目录;查询缓存 _queries.json 也在这里')
    ap.add_argument('--facet-alloc', action='store_true', help='per-facet 名额分配(等价 FACET_ALLOC=1)')
    ap.add_argument('--cutoff-source', choices=['fars', 'human', 'a4s'], default='fars',
                    help='cutoff 口径:fars=paperreview.json;human=matched_166 cdate;a4s=抽取 submission_date')
    ap.add_argument('--gen-queries-only', action='store_true',
                    help='只生成并缓存查询(纯 LLM,不碰 S2/OpenAlex);S2 队列排到前先预热用')
    ap.add_argument('--channel', choices=['api', 'local', 'cito'], default='api',
                    help='主检索通道:api=S2 线上;local=本地 tantivy 快照索引;'
                         'cito=私有混合检索(S2+SPECTER2,自托管无限流)(参数硬切,无降级)')
    ap.add_argument('--oa', choices=['api', 'local', 'off'], default='api',
                    help='第二通道:api=OpenAlex 线上;local=本地 OA 索引;off=关闭')
    ap.add_argument('--dump-pool', action='store_true',
                    help='候选池+全 facet 相似度落盘 _pools/(Block A 消融离线重放用)')
    ap.add_argument('--rerank', choices=['facet', 'embed', 'cross'], default=RERANK_MODE,
                    help='重排口径:facet=bge-m3 per-facet(默认);embed=bge-m3 全局余弦;'
                         'cross=深召回+bge-reranker cross-encoder(领域感知,治词面混淆)')
    a = ap.parse_args()
    CHANNEL, OA_CHANNEL, DUMP_POOL, RERANK_MODE = a.channel, a.oa, a.dump_pool, a.rerank
    EXTRACT_DIR = os.path.abspath(a.extract_dir)
    OUT_DIR = os.path.abspath(a.out_dir)
    QUERY_CACHE = os.path.join(OUT_DIR, '_queries.json')
    CUTOFF_SOURCE = a.cutoff_source
    if a.facet_alloc: FACET_ALLOC = True
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.status:
        status(); sys.exit(0)
    pids = [a.pid] if a.pid else sorted(f[:-5] for f in os.listdir(EXTRACT_DIR) if f.endswith('.json'))
    if a.gen_queries_only:
        qc = load_query_cache()
        n_new = 0
        for i, pid in enumerate(pids):
            d = json.load(open(os.path.join(EXTRACT_DIR, f'{pid}.json')))
            for c in d['contributions']:
                had = f'{pid}/{c["id"]}/v5' in qc
                qs = get_queries(pid, c, qc)
                if not had and qs:
                    n_new += 1
                    print(f'  [{i+1}/{len(pids)}] {pid}/{c["id"]} +{len(qs)} queries', flush=True)
        print(f'QUERIES-DONE new={n_new}', flush=True)
        sys.exit(0)
    t0 = time.time()
    for i, pid in enumerate(pids):
        n = process(pid, a.k)
        if n:
            print(f'[{i+1}/{len(pids)}] {pid} +{n} contribs, elapsed {int(time.time()-t0)}s', flush=True)
    print('ALL DONE', flush=True)
