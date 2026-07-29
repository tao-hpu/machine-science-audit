#!/usr/bin/env python3
"""OpenAlex works 快照 → 本地检索索引(docs/_local-index-task.md 追加节 · 交付物 4)。

单阶段(摘要内联在 abstract_inverted_index,无需像 S2 那样先建 KV):
  works 分片(嵌套 updated_date=*/part_*.gz)→ 还原摘要 + type 过滤 → tantivy 索引。
  每输入分片恰好一次 commit + .done 标记:崩在 commit 前 = 未提交自动丢弃,重跑无重复。

字段对齐 S2 索引(scripts/build_local_index.py):title / abstract(截 800)/ cid(OA W-id)/
date(yyyymmdd)/ cites / doi(剥 https://doi.org/ 前缀,对齐 S2 裸 DOI)。

快照与索引默认都在 WD(内置盘不够,works 索引带摘要可能 200-400GB):
  快照  /Volumes/TONY'S WD/data/openalex_snapshot/works/   (env OA_SNAPSHOT_DIR 覆盖)
  索引  /Volumes/TONY'S WD/data/oa_local_index/            (env OA_LOCAL_INDEX_DIR 覆盖)

用法:
    python3 scripts/build_oa_index.py                # 建索引(可反复重跑,断点续)
    python3 scripts/build_oa_index.py --workers 6
    python3 scripts/build_oa_index.py --status
"""
import glob, gzip, json, os, pathlib, sys, time
from collections import Counter

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
try:
    import orjson as _json
    loads = _json.loads
except ImportError:
    loads = json.loads

SNAP = pathlib.Path(os.environ.get(
    'OA_SNAPSHOT_DIR', "/Volumes/TONY'S WD/data/openalex_snapshot/works"))
OUT = pathlib.Path(os.environ.get(
    'OA_LOCAL_INDEX_DIR', "/Volumes/TONY'S WD/data/oa_local_index"))
IDX = OUT / 'tantivy'
DONE = OUT / '_shards_done'
PREP = OUT / '_prep'
ABS_TRIM = 800

# type 过滤:只保论文类,丢 dataset/peer-review/paratext 等非论文条目(spec 要求,分布写进报告)。
# 白名单法(未知 type 一律丢),比黑名单稳:OA type 词表偶有新增,不让噪声漏进索引。
PAPER_TYPES = {
    'article', 'preprint', 'book', 'book-chapter', 'dissertation',
    'review', 'report', 'letter', 'editorial', 'standard',
    'reference-entry', 'book-series', 'monograph', 'proceedings-article',
}


def shards():
    """OA 快照的全部 .gz 分片(嵌套 updated_date=*/part_*.gz;下载已完整,无 .done 标记)。"""
    return sorted(glob.glob(f'{SNAP}/**/*.gz', recursive=True))


def done_key(f):
    """相对快照根的路径当断点键(part_0000.gz 在多目录重名,必须带目录)。"""
    return str(pathlib.Path(f).relative_to(SNAP)).replace('/', '__')


def schema():
    import tantivy
    sb = tantivy.SchemaBuilder()
    sb.add_text_field('title', stored=True)
    sb.add_text_field('abstract', stored=True)                        # 截 800,建索引
    sb.add_text_field('cid', stored=True, tokenizer_name='raw')       # OpenAlex W-id
    sb.add_integer_field('date', stored=True, indexed=True, fast=True)  # yyyymmdd;缺=0
    sb.add_integer_field('cites', stored=True)
    sb.add_text_field('doi', stored=True, tokenizer_name='raw')
    return sb.build()


def open_index():
    import tantivy
    IDX.mkdir(parents=True, exist_ok=True)
    return tantivy.Index(schema(), path=str(IDX))


def parse_date(r):
    pd, y = r.get('publication_date'), r.get('publication_year')
    if pd:
        try:
            a, m, d = str(pd)[:10].split('-')
            return int(a) * 10000 + int(m) * 100 + int(d)
        except ValueError:
            pass
    if y:
        try:
            return int(y) * 10000 + 101
        except (ValueError, TypeError):
            pass
    return 0


def reconstruct_abstract(aii):
    """abstract_inverted_index {word: [pos,...]} → 原文顺序拼回,截 ABS_TRIM。"""
    if not aii:
        return None
    positions = []
    for word, poss in aii.items():
        for p in poss:
            positions.append((p, word))
    if not positions:
        return None
    positions.sort()
    return ' '.join(w for _, w in positions)[:ABS_TRIM]


def clean_doi(doi):
    if not doi:
        return None
    return str(doi).replace('https://doi.org/', '').replace('http://doi.org/', '') or None


def oa_id(r):
    """https://openalex.org/W123 → W123。"""
    i = r.get('id') or ''
    return i.rsplit('/', 1)[-1] if i else ''


def prep_shard(f):
    """Worker 进程:解压+解析一个 works 分片 → 预制文档批落 _prep/<key>.pkl
    (先 .tmp 再原子改名)。返回 (key, 统计, pkl 路径)。统计含被过滤 type 分布。"""
    import pickle
    key = done_key(f)
    tmp, out = PREP / f'{key}.tmp', PREP / f'{key}.pkl'
    st = {'n': 0, 'with_abstract': 0, 'date0': 0, 'skip_notitle': 0,
          'skip_type': 0, 'dropped_types': Counter()}
    with gzip.open(f, 'rt') as fh, open(tmp, 'wb') as ofh:
        batch = []

        def flush():
            if batch:
                pickle.dump(batch, ofh, protocol=5)

        for line in fh:
            r = loads(line)
            if not r.get('title'):
                st['skip_notitle'] += 1
                continue
            t = r.get('type')
            if t not in PAPER_TYPES:
                st['skip_type'] += 1
                st['dropped_types'][t] += 1
                continue
            date = parse_date(r)
            d = {'title': r['title'], 'cid': oa_id(r), 'date': date,
                 'cites': int(r.get('cited_by_count') or 0)}
            abs_ = reconstruct_abstract(r.get('abstract_inverted_index'))
            if abs_:
                d['abstract'] = abs_
                st['with_abstract'] += 1
            if date == 0:
                st['date0'] += 1
            doi = clean_doi(r.get('doi'))
            if doi:
                d['doi'] = doi
            batch.append(d)
            st['n'] += 1
            if len(batch) >= 50000:
                flush()
                batch = []
        flush()
    st['dropped_types'] = dict(st['dropped_types'])
    tmp.rename(out)
    return key, st, str(out)


def build_index(workers=8):
    """主进程灌 tantivy;workers 个进程并行解析。每分片一次 commit + .done,断点安全。"""
    import pickle, tantivy
    from concurrent.futures import ProcessPoolExecutor, as_completed
    index = open_index()
    writer = index.writer(heap_size=1_500_000_000)
    DONE.mkdir(parents=True, exist_ok=True)
    PREP.mkdir(parents=True, exist_ok=True)
    for junk in PREP.glob('*'):
        junk.unlink()                          # 上次崩溃残留的预制文件全部重做
    files = shards()
    todo = [f for f in files if not (DONE / f'{done_key(f)}.done').exists()]
    print(f'[OA] 分片: {len(files)} 可用, {len(todo)} 待索引, workers={workers}', flush=True)
    stats_f = OUT / '_build_stats.json'
    stats = json.loads(stats_f.read_text()) if stats_f.exists() else \
        {'docs': 0, 'with_abstract': 0, 'date0': 0, 'skip_notitle': 0,
         'skip_type': 0, 'files': 0, 'dropped_types': {}}
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(prep_shard, f): f for f in todo}
        for fut in as_completed(futs):
            key, st, prep = fut.result()
            t0 = time.time()
            with open(prep, 'rb') as fh:
                while True:
                    try:
                        docs = pickle.load(fh)
                    except EOFError:
                        break
                    for d in docs:
                        writer.add_document(tantivy.Document(**d))
            writer.commit()                    # 每分片恰好一次 commit
            (DONE / f'{key}.done').touch()
            os.unlink(prep)
            stats['docs'] += st['n']
            stats['with_abstract'] += st['with_abstract']
            stats['date0'] += st['date0']
            stats['skip_notitle'] += st['skip_notitle']
            stats['skip_type'] += st['skip_type']
            stats['files'] += 1
            dt = stats['dropped_types']
            for k, v in st['dropped_types'].items():
                dt[str(k)] = dt.get(str(k), 0) + v
            stats_f.write_text(json.dumps(stats))
            rate = st['n'] / (time.time() - t0) if time.time() > t0 else 0
            print(f'[OA] ✓ {key} {st["n"]} docs (灌入 {rate:.0f} docs/s), '
                  f'累计 {stats["docs"]} / {stats["files"]} 分片', flush=True)
    el = time.time() - t_start
    print(f'[OA] 索引完成: {stats["docs"]} docs, '
          f'有摘要 {stats["with_abstract"]/max(stats["docs"],1):.1%}, '
          f'无日期 {stats["date0"]/max(stats["docs"],1):.1%}, '
          f'丢弃 type {stats["skip_type"]} / 无题 {stats["skip_notitle"]}, '
          f'用时 {el/3600:.1f}h', flush=True)
    print('OA-INDEX-DONE', flush=True)


def status():
    files = shards()
    ndone = len(list(DONE.glob('*.done'))) if DONE.exists() else 0
    print(f'快照分片: {len(files)} 个 .gz')
    print(f'已索引分片: {ndone}/{len(files)}')
    sf = OUT / '_build_stats.json'
    if sf.exists():
        s = json.loads(sf.read_text())
        print(f'tantivy: {s["files"]} 分片 / {s["docs"]} docs, '
              f'有摘要 {s["with_abstract"]/max(s["docs"],1):.1%}')
        top = sorted(s.get('dropped_types', {}).items(), key=lambda x: -x[1])[:8]
        print(f'丢弃 type top: {top}')
    else:
        print('tantivy: 未开始')


if __name__ == '__main__':
    if '--status' in sys.argv:
        status()
        sys.exit(0)
    if not shards():
        print(f'快照目录无 .gz 分片:{SNAP}(检查 OA_SNAPSHOT_DIR / WD 是否挂载)')
        sys.exit(1)
    w = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 8
    build_index(workers=w)
