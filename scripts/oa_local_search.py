#!/usr/bin/env python3
"""OpenAlex 本地 BM25 检索适配层(docs/_local-index-task.md 追加节 · 交付物 5)。

接口对齐 local_search.search / retrieve_s2_slow.oa_search:
    search(query, cutoff, limit=25) -> [{'title','date','abstract','doi','src':'openalex-local'}, ...]
- BM25 于 title+abstract,取 top-100 后按 date <= cutoff 过滤截到 limit。
- 无日期(date=0)文档过滤时排除(防 post-cutoff 泄漏),排除量记在模块级 EXCLUDED_NODATE。
- date 输出 'YYYY-MM-DD'(下游按字符串前缀比较)。

CLI:
    python3 scripts/oa_local_search.py "good word attack" --cutoff 2010-01-01 [--k 25]
"""
import os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_oa_index import open_index

_index = None
_searcher = None
EXCLUDED_NODATE = 0     # 累计因 date=0 被排除的命中数(统计用)
POOL = 100              # BM25 先取 top-100 再过滤


def _get_searcher():
    global _index, _searcher
    if _searcher is None:
        _index = open_index()
        _index.reload()
        _searcher = _index.searcher()
    return _searcher


def _cutoff_int(cutoff):
    d = re.findall(r'\d+', str(cutoff)[:10])
    return int(d[0]) * 10000 + int(d[1]) * 100 + int(d[2]) if len(d) >= 3 else int(d[0]) * 10000 + 1231


def _fmt_date(v):
    return f'{v // 10000:04d}-{v // 100 % 100:02d}-{v % 100:02d}'


def _first(doc, name, default=None):
    v = doc.get(name)
    return v[0] if v else default


def search(query, cutoff, limit=25):
    global EXCLUDED_NODATE
    s = _get_searcher()
    try:
        q = _index.parse_query_lenient(query, ['title', 'abstract'])[0]
    except AttributeError:
        q = _index.parse_query(re.sub(r'[^\w\s]', ' ', query), ['title', 'abstract'])
    co = _cutoff_int(cutoff)
    out = []
    for _score, addr in s.search(q, POOL).hits:
        doc = s.doc(addr).to_dict()
        date = _first(doc, 'date', 0)
        if date == 0:
            EXCLUDED_NODATE += 1
            continue
        if date > co:
            continue
        out.append({'title': _first(doc, 'title', ''),
                    'date': _fmt_date(date),
                    'abstract': _first(doc, 'abstract', ''),
                    'doi': _first(doc, 'doi'),
                    'citations': _first(doc, 'cites', 0),
                    'src': 'openalex-local'})
        if len(out) >= limit:
            break
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('query')
    ap.add_argument('--cutoff', required=True)
    ap.add_argument('--k', type=int, default=25)
    a = ap.parse_args()
    for r in search(a.query, a.cutoff, a.k):
        print(f"{r['date']}  [{r['citations']:>6}] {r['title']}")
    print(f'(date=0 排除 {EXCLUDED_NODATE} 条)', file=sys.stderr)
