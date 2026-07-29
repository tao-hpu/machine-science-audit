#!/usr/bin/env python3
"""检索通道对齐验证(任务书 docs/_local-index-task.md 交付物 3)——产数据,不下结论。

对 20 条验证批(FA0001-06 的贡献)+ FA0007/C1(seminal 金标准):
- 查询取 data/neighbors_s2/_queries.json 的 {pid}/{cid}/v5(缺则回退 v4 → 基础键);
- cutoff 取 data/fars-a-reviews/data/<pid>/paperreview.json 的 submission_date;
- 每查询检索通道取 20,合并去重(按 cid/标题)成候选池,落
  data/neighbors_<channel>_val/<pid>.json(近邻不重排——重排/在题率扫描主线程做)。
自查两个数:
  ① FA0007/C1 池子是否命中 "Good Word Attacks on Statistical Spam Filters"(Lowd & Meek 2005);
  ② 池子对 data/neighbors_facet_val/ 已选近邻的标题覆盖率(OpenAlex 参照 30%/13%)。

通道(--channel,默认 local):
  local — 本地 BM25 tantivy(local_search);src=s2local;客户端过滤,记 date=0 排除数。
  cito  — 私有混合检索(cito_search);src=cito;服务端 published_before 过滤,记不满额缺口。

用法:python3 scripts/validate_local_channel.py [--channel local|cito]
"""
import argparse, json, os, pathlib, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VAL_DIR = pathlib.Path('data/neighbors_facet_val')
QUERIES = json.load(open('data/neighbors_s2/_queries.json'))
GOLD_TITLE = 'good word attacks on statistical spam filters'

# --- 通道选择:两个 adapter 的 search(query,cutoff,limit) 签名与输出形状一致,可直接换 ---
CHANNELS = {
    'local': ('local_search', 'data/neighbors_local_val', 's2local'),
    'cito':  ('cito_search',   'data/neighbors_cito_val',  'cito'),
}


def norm(t):
    return re.sub(r'[^a-z0-9]+', ' ', (t or '').lower()).strip()


def cutoff_of(pid):
    pr = json.load(open(f'data/fars-a-reviews/data/{pid}/paperreview.json'))
    return pr['submission_date'][:10]


def queries_of(pid, cid):
    return (QUERIES.get(f'{pid}/{cid}/v5') or QUERIES.get(f'{pid}/{cid}/v4')
            or QUERIES.get(f'{pid}/{cid}') or [])


def build_pool(pid, cids):
    """对一篇论文的若干贡献产候选池,返回 contributions 列表(同构 neighbors 文件)。"""
    cutoff = cutoff_of(pid)
    contribs = []
    for cid in cids:
        qs = queries_of(pid, cid)
        seen, pool = set(), []
        for q in qs:
            for r in SEARCH.search(q, cutoff, limit=20):
                k = norm(r['title'])
                if k in seen:
                    continue
                seen.add(k)
                pool.append(r)
        contribs.append({'id': cid, 'n_queries': len(qs),
                         'n_candidates': len(pool), 'neighbors': pool})
        print(f'  {pid}/{cid}: {len(qs)} 查询 → 池 {len(pool)}', flush=True)
    return {'paper_id': pid, 'cutoff': cutoff, 'src': SRC, 'contributions': contribs}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--channel', choices=list(CHANNELS), default='local')
    args = ap.parse_args()
    _mod, _out, SRC = CHANNELS[args.channel]
    SEARCH = __import__(_mod)
    OUT_DIR = pathlib.Path(_out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- 20 条验证批:FA0001-06,贡献清单取自 neighbors_facet_val ---
    coverage_hit, coverage_tot, per_paper = 0, 0, []
    for f in sorted(VAL_DIR.glob('FA*.json')):
        val = json.load(open(f))
        pid = val['paper_id']
        out = build_pool(pid, [c['id'] for c in val['contributions']])
        (OUT_DIR / f'{pid}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
        # 覆盖率:已选近邻标题 ∈ 本地池标题?
        pool_titles = {norm(n['title']) for c in out['contributions'] for n in c['neighbors']}
        sel = [norm(n['title']) for c in val['contributions'] for n in c.get('neighbors', [])]
        hit = sum(1 for t in sel if t in pool_titles)
        per_paper.append((pid, hit, len(sel)))
        coverage_hit += hit
        coverage_tot += len(sel)

    # --- FA0007/C1 seminal 金标准 ---
    g = build_pool('FA0007', ['C1'])
    (OUT_DIR / 'FA0007.json').write_text(json.dumps(g, ensure_ascii=False, indent=1))
    gold_hit = any(GOLD_TITLE in norm(n['title'])
                   for n in g['contributions'][0]['neighbors'])

    summary = {
        'channel': args.channel,
        'gold_FA0007_C1_lowd_meek_2005': gold_hit,
        'coverage_overall': f'{coverage_hit}/{coverage_tot}'
                            f' = {coverage_hit / max(coverage_tot, 1):.1%}',
        'coverage_per_paper': [f'{p}: {h}/{t}' for p, h, t in per_paper],
        # local:客户端按 date=0 排除的命中数;cito:服务端过滤,记最后一次不满额缺口
        'excluded_nodate_hits': getattr(SEARCH, 'EXCLUDED_NODATE', None),
        'last_underfill': getattr(SEARCH, 'LAST_UNDERFILL', None),
        'corpus_release': getattr(SEARCH, 'LAST_CORPUS_RELEASE', None),
        # hybrid 融合 provenance:fused=true 的查询数 / 降级为纯语义的查询数
        'fused_true': getattr(SEARCH, 'FUSED_TRUE', None),
        'fused_false': getattr(SEARCH, 'FUSED_FALSE', None),
        'elapsed_s': round(time.time() - t0, 1),
    }
    (OUT_DIR / '_validation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    print('VALIDATE-LOCAL-DONE', flush=True)
