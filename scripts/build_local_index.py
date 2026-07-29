#!/usr/bin/env python3
"""S2 快照 → 本地检索索引(任务书 docs/_local-index-task.md 交付物 1)。

两阶段,均断点可续,只处理下载器已打 .done 标记的分片:
  A. abstracts 分片 → sqlite KV(corpusid → abstract[:800])→ data/s2_local_index/abstracts.db
  B. papers 分片 join KV → tantivy 索引 → data/s2_local_index/tantivy/
     每输入文件恰好一次 commit + .done 标记:崩在 commit 前 = 未提交自动丢弃,重跑无重复。

用法:
    python3 scripts/build_local_index.py             # A 完成后自动进 B(要求快照下载完毕)
    python3 scripts/build_local_index.py --status
"""
import glob, gzip, json, os, pathlib, sqlite3, sys, time

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

# 路径可用 .env / 环境变量覆盖(快照或索引放外置盘时改这两个):
SNAP = pathlib.Path(os.environ.get('S2_SNAPSHOT_DIR', 'data/s2_snapshot'))
OUT = pathlib.Path(os.environ.get('S2_LOCAL_INDEX_DIR', 'data/s2_local_index'))
DB = OUT / 'abstracts.db'
IDX = OUT / 'tantivy'
DONE_B = OUT / '_papers_done'
ABS_TRIM = 800


def shards(ds):
    """某数据集已完整下载(有 .done)的分片列表。"""
    return sorted(p for p in glob.glob(f'{SNAP}/{ds}/*.gz')
                  if pathlib.Path(f'{p}.done').exists())


def snapshot_complete(ds):
    mf = SNAP / ds / '_manifest.json'
    if not mf.exists():
        return False
    return len(shards(ds)) == len(json.loads(mf.read_text()))


# ---------------- 阶段 A:abstracts → sqlite KV ----------------

def db_conn():
    OUT.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA synchronous=NORMAL')
    con.execute('CREATE TABLE IF NOT EXISTS kv (cid INTEGER PRIMARY KEY, abstract TEXT)')
    con.execute('CREATE TABLE IF NOT EXISTS done_files (name TEXT PRIMARY KEY)')
    return con


def build_kv():
    con = db_conn()
    done = {r[0] for r in con.execute('SELECT name FROM done_files')}
    files = shards('abstracts')
    todo = [f for f in files if pathlib.Path(f).name not in done]
    print(f'[A] abstracts 分片: {len(files)} 可用, {len(todo)} 待处理', flush=True)
    for f in todo:
        t0, n, batch = time.time(), 0, []
        with gzip.open(f, 'rt') as fh:
            for line in fh:
                r = loads(line)
                cid, abs_ = r.get('corpusid'), r.get('abstract')
                if cid is None or not abs_:
                    continue
                batch.append((int(cid), abs_[:ABS_TRIM]))
                n += 1
                if len(batch) >= 50000:
                    con.executemany('INSERT OR REPLACE INTO kv VALUES (?,?)', batch)
                    batch = []
        if batch:
            con.executemany('INSERT OR REPLACE INTO kv VALUES (?,?)', batch)
        if n == 0:
            raise RuntimeError(f'{f} 解析出 0 行摘要,分片格式与预期不符,停下检查')
        con.execute('INSERT OR REPLACE INTO done_files VALUES (?)', (pathlib.Path(f).name,))
        con.commit()
        print(f'[A] ✓ {pathlib.Path(f).name} {n} 行 ({n/(time.time()-t0):.0f} rec/s)', flush=True)
    total = con.execute('SELECT COUNT(*) FROM kv').fetchone()[0]
    print(f'[A] KV 完成: {total} 条摘要', flush=True)
    con.close()
    return total


# ---------------- 阶段 B:papers join KV → tantivy ----------------

def schema():
    import tantivy
    sb = tantivy.SchemaBuilder()
    sb.add_text_field('title', stored=True)
    sb.add_text_field('abstract', stored=True)                       # 截 800,建索引
    sb.add_text_field('cid', stored=True, tokenizer_name='raw')
    sb.add_integer_field('date', stored=True, indexed=True, fast=True)  # yyyymmdd;缺=0
    sb.add_integer_field('cites', stored=True)
    sb.add_text_field('doi', stored=True, tokenizer_name='raw')
    return sb.build()


def open_index():
    import tantivy
    IDX.mkdir(parents=True, exist_ok=True)
    return tantivy.Index(schema(), path=str(IDX))


def parse_date(r):
    pd, y = r.get('publicationdate'), r.get('year')
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


PREP = OUT / '_prep'


def _lookup(cur, cids):
    out = {}
    for i in range(0, len(cids), 900):
        sub = cids[i:i + 900]
        q = f"SELECT cid, abstract FROM kv WHERE cid IN ({','.join('?' * len(sub))})"
        out.update(cur.execute(q, sub).fetchall())
    return out


def prep_shard(f):
    """Worker 进程:解压+解析一个 papers 分片 + join 摘要 KV,预制文档批
    落 _prep/<name>.pkl(先 .tmp 再原子改名)。返回 (name, 统计, pkl 路径)。"""
    import pickle
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    cur = con.cursor()
    name = pathlib.Path(f).name
    tmp, out = PREP / f'{name}.tmp', PREP / f'{name}.pkl'
    st = {'n': 0, 'with_abstract': 0, 'date0': 0}
    with gzip.open(f, 'rt') as fh, open(tmp, 'wb') as ofh:
        batch = []

        def emit(recs):
            absmap = _lookup(cur, [c for c, _ in recs])
            docs = []
            for cid, r in recs:
                date = parse_date(r)
                d = {'title': r['title'], 'cid': str(cid), 'date': date,
                     'cites': int(r.get('citationcount') or 0)}
                abs_ = absmap.get(cid)
                if abs_:
                    d['abstract'] = abs_
                    st['with_abstract'] += 1
                if date == 0:
                    st['date0'] += 1
                doi = (r.get('externalids') or {}).get('DOI')
                if doi:
                    d['doi'] = str(doi)
                docs.append(d)
                st['n'] += 1
            pickle.dump(docs, ofh, protocol=5)

        for line in fh:
            r = loads(line)
            if not r.get('title') or r.get('corpusid') is None:
                continue
            batch.append((int(r['corpusid']), r))
            if len(batch) >= 50000:
                emit(batch)
                batch = []
        if batch:
            emit(batch)
    con.close()
    tmp.rename(out)
    return name, st, str(out)


def build_index(workers=8):
    """主进程只灌 tantivy;workers 个进程并行做解析+KV join(M4 Max 吃得下)。
    断点语义不变:每分片一次 commit + .done;崩在 commit 前 = 自动丢弃重做。"""
    import pickle, tantivy
    from concurrent.futures import ProcessPoolExecutor, as_completed
    index = open_index()
    writer = index.writer(heap_size=1_500_000_000)
    DONE_B.mkdir(parents=True, exist_ok=True)
    PREP.mkdir(parents=True, exist_ok=True)
    for junk in PREP.glob('*'):
        junk.unlink()                          # 上次崩溃残留的预制文件全部重做
    files = shards('papers')
    todo = [f for f in files if not (DONE_B / f'{pathlib.Path(f).name}.done').exists()]
    print(f'[B] papers 分片: {len(files)} 可用, {len(todo)} 待索引, workers={workers}', flush=True)
    stats_f = OUT / '_build_stats.json'
    stats = json.loads(stats_f.read_text()) if stats_f.exists() else \
        {'docs': 0, 'with_abstract': 0, 'date0': 0, 'files': 0}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(prep_shard, f) for f in todo]
        for fut in as_completed(futs):
            name, st, prep = fut.result()
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
            (DONE_B / f'{name}.done').touch()
            os.unlink(prep)
            stats['docs'] += st['n']
            stats['with_abstract'] += st['with_abstract']
            stats['date0'] += st['date0']
            stats['files'] += 1
            stats_f.write_text(json.dumps(stats))
            print(f'[B] ✓ {name} {st["n"]} docs (灌入 {st["n"]/(time.time()-t0):.0f} docs/s), '
                  f'累计 {stats["docs"]}', flush=True)
    print(f'[B] 索引完成: {stats["docs"]} docs, '
          f'有摘要 {stats["with_abstract"]/max(stats["docs"],1):.1%}, '
          f'无日期 {stats["date0"]/max(stats["docs"],1):.1%}', flush=True)
    print('LOCAL-INDEX-DONE', flush=True)


def status():
    for ds in ('abstracts', 'papers'):
        mf = SNAP / ds / '_manifest.json'
        tot = len(json.loads(mf.read_text())) if mf.exists() else '?'
        print(f'快照 {ds}: {len(shards(ds))}/{tot} 分片就绪')
    if DB.exists():
        con = sqlite3.connect(DB)
        print(f'KV: {con.execute("SELECT COUNT(*) FROM done_files").fetchone()[0]} 分片入库, '
              f'{con.execute("SELECT COUNT(*) FROM kv").fetchone()[0]} 条摘要')
        con.close()
    else:
        print('KV: 未开始')
    sf = OUT / '_build_stats.json'
    if sf.exists():
        s = json.loads(sf.read_text())
        print(f'tantivy: {s["files"]} 分片 / {s["docs"]} docs')
    else:
        print('tantivy: 未开始')


if __name__ == '__main__':
    if '--status' in sys.argv:
        status()
        sys.exit(0)
    if not (snapshot_complete('abstracts') and snapshot_complete('papers')):
        print('快照尚未下载完整(用 download_s2_snapshot.py --status 查),先跑完再建索引。')
        sys.exit(1)
    w = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 8
    build_kv()
    build_index(workers=w)
