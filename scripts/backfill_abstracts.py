#!/usr/bin/env python3
"""近邻空摘要回填:S2/OpenAlex 记录 ~25%(v6)/11.5%(v5)无摘要,裁判只能看标题猜,
是判定分歧温床。用 DOI 走 OpenAlex 单篇端点回填(不占 S2 配额);无 DOI 的按标题搜索匹配。

用法:
    python3 scripts/backfill_abstracts.py data/neighbors_facet_val   # 指定目录
    python3 scripts/backfill_abstracts.py --dry data/neighbors_s2    # 只统计不写
原地改写 JSON;回填标 src 后缀 '+oa-backfill'。断点天然:有摘要的跳过。
"""
import json, glob, re, sys, time, urllib.request, urllib.parse

ROOT = __file__.rsplit('/', 2)[0]
ENV = {}
for l in open(f'{ROOT}/.env'):
    if '=' in l and not l.strip().startswith('#'):
        k, v = l.split('=', 1); ENV[k.strip()] = v.strip().strip('"').strip("'")

UA = {'User-Agent': 'novelty-audit/0.3 (mailto:tan1@my.hpu.edu)'}


def http(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def deinvert(inv):
    if not inv: return ''
    pos = [(p, w) for w, ps in inv.items() for p in ps]
    return ' '.join(w for _, w in sorted(pos))


def arxiv_abs(doi):
    """arXiv DOI (10.48550/arXiv.XXXX.YYYYY) 走 arXiv 官方 API,权威且免 OpenAlex 记录污染。"""
    m = re.search(r'10\.48550/arxiv\.(.+)$', doi, re.I)
    if not m: return ''
    try:
        req = urllib.request.Request(f'http://export.arxiv.org/api/query?id_list={m.group(1)}', headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            xml = r.read().decode()
        mm = re.search(r'<summary>(.*?)</summary>', xml, re.S)
        return re.sub(r'\s+', ' ', mm.group(1)).strip() if mm else ''
    except Exception:
        return ''


def crossref_abs(doi):
    try:
        w = http(f'https://api.crossref.org/works/{urllib.parse.quote(doi)}')['message']
        ab = w.get('abstract') or ''
        return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', ab)).strip()
    except Exception:
        return ''


_OA_KEY_DEAD = [False]
def _oa_params(p):
    """2026-07-08:premium key 配额耗尽时 OpenAlex 对带 key 请求一律 429,无 key
    polite pool 正常 —— key 一旦 429 永久降级(与 retrieve_s2_slow 同教训)。"""
    p = dict(p); p['mailto'] = 'tan1@my.hpu.edu'
    if ENV.get('OPENALEX_API_KEY') and not _OA_KEY_DEAD[0]:
        p['api_key'] = ENV['OPENALEX_API_KEY']
    return p

def _oa_get(url_base, p):
    p = _oa_params(p)
    try:
        return http(f'{url_base}?{urllib.parse.urlencode(p)}')
    except urllib.error.HTTPError as e:
        if e.code == 429 and 'api_key' in p:
            _OA_KEY_DEAD[0] = True
            p.pop('api_key')
            try:
                return http(f'{url_base}?{urllib.parse.urlencode(p)}')
            except Exception:
                return None
        return None
    except Exception:
        return None

def oa_by_doi(doi):
    """OpenAlex 兜底。已知风险:个别记录 title 对但 abstract 被污染成别的论文
    (实测 H2O 2306.14048 挂着 Hyde-IKV 摘要),故仅作 arXiv/Crossref 之后的最后备选。"""
    w = _oa_get(f'https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}',
                {'select': 'abstract_inverted_index'})
    return deinvert(w.get('abstract_inverted_index')) if w else ''


def norm(t):
    return re.sub(r'[^a-z0-9]', '', t.lower())


def oa_by_title(title):
    res = _oa_get('https://api.openalex.org/works',
                  {'filter': f'title.search:{title[:200]}', 'per-page': '3',
                   'select': 'title,abstract_inverted_index'})
    for w in (res or {}).get('results', []):
        if norm(w.get('title') or '') == norm(title):
            return deinvert(w.get('abstract_inverted_index'))
    return ''


def main():
    import os
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    dry = '--dry' in sys.argv
    targets = args or ['data/neighbors_facet_val']
    files = []
    for t in targets:  # 目录或单个 json 文件都行(流水线驱动按文件送)
        if os.path.isdir(t):
            files += [f for f in glob.glob(f'{t}/*.json') if not os.path.basename(f).startswith('_')]
        else:
            files.append(t)
    tot = miss = filled = 0
    for f in sorted(files):
        j = json.load(open(f))
        changed = False
        for c in j['contributions']:
            for n in c.get('neighbors') or []:
                tot += 1
                if (n.get('abstract') or '').strip():
                    continue
                miss += 1
                if dry: continue
                doi = re.sub(r'^https?://doi.org/', '', n.get('doi') or '')
                ab, via = '', ''
                if doi:
                    for fn, tag in ((arxiv_abs, 'arxiv'), (crossref_abs, 'crossref'), (oa_by_doi, 'oa')):
                        ab = fn(doi)
                        if len(ab) >= 80: via = tag; break
                        ab = ''
                if not ab:
                    ab = oa_by_title(n.get('title') or ''); via = 'oa-title'
                time.sleep(0.15)
                if len(ab) >= 80:
                    n['abstract'] = ab[:800]
                    n['src'] = (n.get('src') or '') + f'+backfill-{via}'
                    filled += 1; changed = True
        if changed:
            json.dump(j, open(f, 'w'), ensure_ascii=False, indent=1)
            print(f'{f}: saved', flush=True)
    print(f'BACKFILL-DONE total={tot} missing={miss} filled={filled}', flush=True)


if __name__ == '__main__':
    main()
