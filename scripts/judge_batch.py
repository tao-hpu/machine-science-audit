#!/usr/bin/env python3
"""FARS 全量判定批跑器 —— 双主裁判 + facet 级分歧仲裁,断点续跑,后台可挂。

对 data/extractions/ 里每篇每条贡献,取 data/neighbors_s2/ 的近邻,过 judge_v3(尺子 v7):
两主裁判(gpt-4o / claude-sonnet-4-6,temperature=0)全量判;facet 级覆盖分歧时调
gemini-2.5-pro 仲裁(2/3 多数票);state 由共识 facet 确定性导出(consensus_state_v2)。
协议于 2026-07-07 在 20 条验证批上定案:split 清零,gemini 仲裁站边 10:5 偏 sonnet。
每条贡献一落盘(data/judgments_full/<PID>.json),已判的跳过 → 断点续跑、崩溃不丢。
旧 state 级多数票仍记 consensus_state 供对照。

用法:
    python3 scripts/judge_batch.py            # 全量(续跑)
    python3 scripts/judge_batch.py --status   # 看进度
    python3 scripts/judge_batch.py --models gpt-4o,claude-sonnet-4-6   # 指定主裁判
    # 换语料(人类基线):
    python3 scripts/judge_batch.py --ext data/extractions_human \
        --nbr data/neighbors_human --out data/judgments_human
    # 分片并行(按 paper 切,i 从 1 数;两片各开一个进程,输出文件不冲突):
    python3 scripts/judge_batch.py --shard 1/2 &  python3 scripts/judge_batch.py --shard 2/2 &
    python3 scripts/judge_batch.py --pids FA0001,FA0007   # 只判指定 paper
"""
import json, os, sys, time, pathlib
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import scripts.judge_v3 as J
from scripts.facet_consensus import covset, derive_state, TIEBREAK

# 默认 FARS;main() 里 --ext/--nbr/--out 覆盖
EXT = pathlib.Path('data/extractions')
NBR = pathlib.Path('data/neighbors_s2')
OUT = pathlib.Path('data/judgments_full')
DEFAULT_MODELS = 'gpt-4o,claude-sonnet-4-6'


def ext_files():
    """按文件名排序的抽取文件;跳过 _ 开头的辅助文件(如 _queries.json 这类)。"""
    return sorted(f for f in EXT.glob('*.json') if not f.name.startswith('_'))


def neighbors_for(pid, cid):
    """只返回检索已完成(s2_complete)的贡献近邻;半成品返回 None 由调用方跳过。"""
    j = json.load(open(NBR / f'{pid}.json'))
    c = next((c for c in j['contributions'] if c['id'] == cid), None)
    if not c or not c.get('s2_complete'):
        return None
    return c['neighbors']


def consensus(states):
    """多数定态;平票或全空记 split。"""
    votes = [s for s in states if s]
    if not votes:
        return None
    top = Counter(votes).most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return 'split'
    return top[0][0]


def already_done(pid):
    f = OUT / f'{pid}.json'
    if not f.exists():
        return set()
    try:
        return {c['cid'] for c in json.load(open(f))['contributions']}
    except Exception:
        return set()


def save(pid, title, rows):
    json.dump({'paper_id': pid, 'title': title, 'contributions': rows},
              open(OUT / f'{pid}.json', 'w'), ensure_ascii=False, indent=1)


def status(models):
    ext = ext_files()
    total_c = done_c = 0
    state_tally = Counter()
    for f in ext:
        e = json.load(open(f))
        pid = e['paper_id']
        if (NBR / f'{pid}.json').exists():
            nj = json.load(open(NBR / f'{pid}.json'))
            total_c += sum(1 for c in nj['contributions'] if c.get('s2_complete'))
        d = OUT / f'{pid}.json'
        if d.exists():
            for c in json.load(open(d))['contributions']:
                done_c += 1
                state_tally[c.get('consensus_state_v2') or c.get('consensus_state')] += 1
    print(f'judged contributions: {done_c}/{total_c} (papers with neighbors only)')
    if state_tally:
        print('consensus_v2:', dict(state_tally))


def main():
    global EXT, NBR, OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=DEFAULT_MODELS)
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='只判前 N 条贡献(验证用,0=全量)')
    ap.add_argument('--ext', default=str(EXT), help='抽取目录')
    ap.add_argument('--nbr', default=str(NBR), help='近邻目录')
    ap.add_argument('--out', default=str(OUT), help='判定输出目录')
    ap.add_argument('--pids', default='', help='只判这些 paper(逗号分隔)')
    ap.add_argument('--shard', default='', help='i/n:按排序后的 paper 列表取第 i 片(1 起数);'
                                               '分片粒度是 paper,保证输出文件不被两个进程同写')
    a = ap.parse_args()
    models = a.models.split(',')
    EXT, NBR, OUT = pathlib.Path(a.ext), pathlib.Path(a.nbr), pathlib.Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.status:
        status(models)
        return
    ext = ext_files()
    if a.pids:
        want = set(a.pids.split(','))
        ext = [f for f in ext if f.stem in want]
    if a.shard:
        i, n = map(int, a.shard.split('/'))
        assert 1 <= i <= n, f'--shard {a.shard}: i 必须在 1..n'
        ext = [f for j, f in enumerate(ext) if j % n == i - 1]
    n_judged = 0
    for f in ext:
        e = json.load(open(f))
        pid = e['paper_id']
        if not (NBR / f'{pid}.json').exists():
            continue
        done = already_done(pid)
        existing = []
        if (OUT / f'{pid}.json').exists():
            existing = json.load(open(OUT / f'{pid}.json'))['contributions']
        rows = list(existing)
        changed = False
        for c in e['contributions']:
            if c['id'] in done:
                continue
            nbrs = neighbors_for(pid, c['id'])
            if not nbrs:
                continue
            per_model = {}
            for m in models:
                try:
                    r = J.judge_one(m, c, nbrs, keep_raw=True)
                    per_model[m] = {'state': r['state'], 'facets': r['facets'],
                                    'self_check': r['self_check_triggered'],
                                    'raw': r['raw']}
                except Exception as ex:
                    per_model[m] = {'state': None, 'facets': None, 'error': repr(ex)[:100]}
                time.sleep(0.5)
            # facet 级分歧 → 第三裁判仲裁
            judges = list(models)
            if all(per_model[m].get('facets') for m in models) and len(models) == 2:
                disagreed = [fa for fa in J.FACETS
                             if bool(covset(per_model[models[0]]['facets'], fa))
                             != bool(covset(per_model[models[1]]['facets'], fa))]
                if disagreed:
                    try:
                        r = J.judge_one(TIEBREAK, c, nbrs, keep_raw=True)
                        per_model[TIEBREAK] = {'state': r['state'], 'facets': r['facets'],
                                               'self_check': r['self_check_triggered'], 'raw': r['raw']}
                    except Exception as ex:
                        per_model[TIEBREAK] = {'state': None, 'facets': None, 'error': repr(ex)[:100]}
                    if per_model[TIEBREAK].get('facets'):
                        judges.append(TIEBREAK)
                    time.sleep(0.5)
            # facet 级共识 + 确定性 state
            cs2, cons_facets, unresolved = None, None, []
            voters = [m for m in judges if per_model[m].get('facets')]
            if len(voters) >= 2:
                cons = {}
                for fa in J.FACETS:
                    votes = [covset(per_model[m]['facets'], fa) for m in voters]
                    covered = [v for v in votes if v]
                    if len(covered) * 2 > len(votes):
                        cons[fa] = set().union(*covered)
                    elif len(covered) * 2 < len(votes):
                        cons[fa] = set()
                    else:
                        unresolved.append(fa); cons[fa] = set()
                cs2 = 'split' if unresolved else derive_state(cons)
                cons_facets = {fa: sorted(cons[fa]) if cons[fa] else None for fa in J.FACETS}
            cs = consensus([per_model[m]['state'] for m in models])
            row = {'cid': c['id'], 'type': c.get('type'), 'prompt_version': J.PROMPT_VERSION,
                   'consensus_state': cs, 'consensus_state_v2': cs2,
                   'consensus_facets': cons_facets, 'per_model': per_model}
            if unresolved:
                row['consensus_unresolved'] = unresolved
            rows.append(row)
            changed = True
            save(pid, e['title'], rows)
            print(f'{pid}/{c["id"]:4s} -> {str(cs2):14s} ' +
                  ' '.join(f'{m.split("-")[0]}:{per_model[m]["state"]}' for m in per_model), flush=True)
            n_judged += 1
            if a.limit and n_judged >= a.limit:
                print(f'LIMIT {a.limit} reached', flush=True)
                return
        if changed:
            print(f'  [{pid}] done ({len(rows)} contribs)', flush=True)
    print('BATCH-DONE', flush=True)


if __name__ == '__main__':
    main()
