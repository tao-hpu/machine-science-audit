#!/usr/bin/env python3
"""cito 私有混合检索适配层(检索通道 --channel cito)。

cito = 自托管语义/关键词/混合检索,S2 ~146M 篇 + SPECTER2 向量,快照冻结 2026-06-24。
建来替掉纯 BM25 本地通道:语义召回能把对题先例捞进池(本地 BM25 词面匹配捞不到的),
且自托管无 1 req/s 限流、快照可冻结 = 质量/无限流/可复现三样一起拿。

接口对齐 retrieve_s2_slow.s2_search / local_search.search:
    search(query, cutoff, limit=20) -> [{'title','date','abstract','doi','citations','src':'cito'}, ...]

两个关键设计(与基线对齐,别乱改):
- **摘要默认截 800**:3.37 基线(s2_search)在 800 字符摘要上判分。默认截 800 隔离出
  纯检索质量;--full-abstract 关掉截断,单独测「全文摘要」的增益(cito 原生给 1100-1700)。
- **cutoff = published_before(严格 <)**:服务端在裁剪到 limit 之前过滤,靶子当天及之后一律砍,
  未来论文不占名额(防泄漏)。粒度回退/无日期由服务端按约定处理(同年默认砍),客户端不再过滤。

服务端限制(见 docs/cito_change_response.md 回执):
- limit 上限 100;重过滤场景可能不满额(存活 < limit),模块级 LAST_UNDERFILL 记录。
- date_granularity: 'day'/'year'/None(S2 无月粒度)。年粒度约占 22%,「同年砍」是实起作用的规则。

CLI:
    python3 scripts/cito_search.py "good word attack" --cutoff 2010-01-01 [--k 20] [--full-abstract]
"""
import os, re, sys, time, json, urllib.parse, urllib.request, urllib.error

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE = os.environ.get('CITO_API_BASE', 'https://api.cito.fim.ai').rstrip('/')
KEY = os.environ.get('CITO_API_KEY', '')
ABS_TRIM = 800          # 对齐 s2_search 基线;--full-abstract 置 None
MAX_LIMIT = 100         # 服务端上限
TIMEOUT = 30
RETRIES = 4

LAST_UNDERFILL = 0      # 上次调用「请求 limit - 实际返回」的缺口(>0 = 重过滤下池子被抽干)
LAST_CORPUS_RELEASE = None   # 服务端回显的快照版本,盖戳用
# 跨域查询扩展(方案 #1,2026-07-12 服务端上线):expand=true 时服务端用 LLM(temperature=0)
# 把概念查询改写成邻近领域 term-of-art,多路检索 RRF 融合。响应回显 expansion=[变体...] 供审计
# 做 provenance;expansion=None 表示未生效、等同普通检索。每查约 +1s(服务端有缓存)。
LAST_EXPANSION = None
# hybrid 融合 provenance:fused=true 表示该次 BM25 确实参与 RRF,false=降级为纯语义。
# tantivy 索引 2026-07-09 才上线,之前 hybrid 静默降级 dense-only,所以要断言。
FUSED_TRUE = 0
FUSED_FALSE = 0
LAST_FUSED = None


def _norm_cutoff(cutoff):
    """任意 cutoff → 严格 YYYY-MM-DD(cito 对格式严格,422 on 2017-6-1)。
    只给年 → YYYY-01-01(泄漏安全:宁可整年砍掉也不放未来论文进来)。None → None(不过滤)。"""
    if not cutoff:
        return None
    d = re.findall(r'\d+', str(cutoff))
    if not d:
        return None
    y = int(d[0])
    mo = int(d[1]) if len(d) >= 2 else 1
    da = int(d[2]) if len(d) >= 3 else 1
    if len(d) < 3:
        print(f'[cito] cutoff {cutoff!r} 无日粒度 → 用 {y:04d}-01-01(整年砍,泄漏安全)', file=sys.stderr)
    return f'{y:04d}-{mo:02d}-{da:02d}'


def _get(params):
    url = f'{BASE}/search?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {KEY}'} if KEY else {})
    last = None
    for i in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode('utf-8', 'replace')
            if e.code == 422:                       # 参数错,重试无意义
                raise RuntimeError(f'cito 422(参数非法): {body}')
            if e.code in (429, 500, 502, 503, 504):
                last = f'{e.code} {body}'
                time.sleep(1.5 * (i + 1))
                continue
            raise RuntimeError(f'cito {e.code}: {body}')
        except (urllib.error.URLError, TimeoutError) as e:
            last = str(e)
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f'cito 连续 {RETRIES} 次失败: {last}')


def search(query, cutoff, limit=20, mode='hybrid', full_abstract=False, expand=False):
    global LAST_UNDERFILL, LAST_CORPUS_RELEASE, LAST_FUSED, FUSED_TRUE, FUSED_FALSE, LAST_EXPANSION
    k = min(limit, MAX_LIMIT)
    if limit > MAX_LIMIT:
        print(f'[cito] limit {limit} 超服务端上限 {MAX_LIMIT},截到 {MAX_LIMIT}', file=sys.stderr)
    params = {'q': query, 'mode': mode, 'limit': k, 'enrich': 'true'}
    if expand:
        params['expand'] = 'true'      # 跨域查询扩展,见 LAST_EXPANSION 注释
    pb = _norm_cutoff(cutoff)
    if pb:
        params['published_before'] = pb
    data = _get(params)
    LAST_CORPUS_RELEASE = data.get('corpus_release')
    LAST_EXPANSION = data.get('expansion')     # [变体...] 或 None(未生效)
    LAST_FUSED = data.get('fused')          # hybrid 下 true=BM25 参与融合,false=降级纯语义
    if mode == 'hybrid':
        if LAST_FUSED is True:
            FUSED_TRUE += 1
        elif LAST_FUSED is False:
            FUSED_FALSE += 1
    hits = data.get('hits', [])
    LAST_UNDERFILL = max(0, k - len(hits))
    trim = None if full_abstract else ABS_TRIM
    out = []
    for h in hits:
        if not h.get('title'):
            continue
        ab = h.get('abstract') or ''
        out.append({'title': h['title'],
                    'date': h.get('publication_date'),       # 'YYYY-MM-DD' / 'YYYY' / None
                    'abstract': ab[:trim] if trim else ab,
                    'doi': h.get('doi'),
                    'citations': h.get('cites'),
                    'src': 'cito'})
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('query')
    ap.add_argument('--cutoff', default=None)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--mode', default='hybrid', choices=['hybrid', 'semantic', 'keyword'])
    ap.add_argument('--full-abstract', action='store_true')
    ap.add_argument('--expand', action='store_true', help='服务端跨域查询扩展(方案 #1)')
    a = ap.parse_args()
    rows = search(a.query, a.cutoff, a.k, a.mode, a.full_abstract, a.expand)
    for r in rows:
        print(f"{str(r['date'] or '????'):<10}  [{(r['citations'] or 0):>6}] {r['title'][:80]}")
    if LAST_EXPANSION:
        print(f'[expansion] {LAST_EXPANSION}', file=sys.stderr)
    print(f'(返回 {len(rows)} 条; 缺口 {LAST_UNDERFILL}; 快照 {LAST_CORPUS_RELEASE})', file=sys.stderr)
