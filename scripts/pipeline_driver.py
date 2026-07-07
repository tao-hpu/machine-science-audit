#!/usr/bin/env python3
"""流水线驱动:检索(慢,S2 429 磨)还在跑时,把「已收尾的 paper」提前送进回填+判定。

背景(2026-07-07 通宵档):P1 v6 重选实测 ~272s/贡献(S2 search 端点 429 风暴,
比计划的 3-6h 慢一个量级,全量 ~41h)。判定若干等检索收尾,LLM 车道空转两天。
本驱动按 paper 粒度流水:closed(抽取里每条贡献都 s2_complete)→ 回填摘要 → 判定。

安全前提(都已验证):
- retrieve_s2_slow 对 closed paper 不再写文件(done 全跳过,todo 空直接 return,写都不写);
- backfill_abstracts 幂等(有摘要跳过);judge_batch 断点续跑(已判贡献跳过)。
驱动绝不碰半成品文件,与检索进程无写冲突;驱动自身串行(回填完才判定)。

错误回补:每轮判定后扫输出,consensus_state_v2 为空或两主裁判都 error 的行删掉重判,
每条贡献最多回补 2 次(防持续故障烧钱),状态记 <out>/_driver_state.json。

检索日志出现 ALL DONE 后:最后一轮全量回填+判定(含半满残条,judge 自己跳过
非 s2_complete 的贡献),然后退出并打 DRIVER-DONE。

用法:
  nohup python3 -u scripts/pipeline_driver.py \
      --ext data/extractions --nbr data/neighbors_s2_v6 --out data/judgments_full \
      --retrieve-log logs/retrieve_v6.log >> logs/pipeline_v6.log 2>&1 &
"""
import argparse, json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.chdir(ROOT)
PY = sys.executable or 'python3'


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def closed_papers(ext, nbr):
    """抽取里每条贡献在近邻文件里都 s2_complete 的 paper(检索不会再写它)。"""
    out = []
    for f in sorted(p for p in ext.glob('*.json') if not p.name.startswith('_')):
        pid = f.stem
        e, n = jload(f), jload(nbr / f'{pid}.json')
        if not e or not n:
            continue
        want = {c['id'] for c in e['contributions']}
        got = {c['id'] for c in n['contributions'] if c.get('s2_complete')}
        if want and want <= got:
            out.append(pid)
    return out


def run(cmd):
    print(f'[driver] $ {" ".join(cmd)}', flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def repair_errors(outdir, state):
    """删掉判废的行(v2 共识为空 / 两主裁判都 error),让 judge 续跑回补;每条最多 2 次。"""
    retried = state.setdefault('repair_count', {})
    n_del = 0
    for f in sorted(outdir.glob('*.json')):
        if f.name.startswith('_'):
            continue
        j = jload(f)
        if not j:
            continue
        keep, changed = [], False
        for row in j['contributions']:
            pm = row.get('per_model') or {}
            mains = [v for v in pm.values()]
            all_err = mains and all(v.get('error') for v in mains)
            bad = all_err or not row.get('consensus_state_v2')
            key = f'{j["paper_id"]}/{row["cid"]}'
            if bad and retried.get(key, 0) < 2:
                retried[key] = retried.get(key, 0) + 1
                changed = True
                n_del += 1
                continue
            keep.append(row)
        if changed:
            j['contributions'] = keep
            json.dump(j, open(f, 'w'), ensure_ascii=False, indent=1)
    if n_del:
        print(f'[driver] repair: deleted {n_del} bad rows for re-judge', flush=True)
    return n_del


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ext', required=True)
    ap.add_argument('--nbr', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--retrieve-log', required=True)
    ap.add_argument('--interval', type=int, default=1200, help='轮询间隔秒(默认 20min)')
    a = ap.parse_args()
    ext, nbr, outdir = pathlib.Path(a.ext), pathlib.Path(a.nbr), pathlib.Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    state_f = outdir / '_driver_state.json'
    state = jload(state_f) or {'backfilled': [], 'judged': []}

    while True:
        retrieval_done = False
        try:
            retrieval_done = 'ALL DONE' in open(a.retrieve_log).read()
        except Exception:
            pass

        closed = closed_papers(ext, nbr)
        todo = [p for p in closed if p not in state['judged']]
        if todo:
            print(f'[driver] {time.strftime("%m-%d %H:%M")} closed={len(closed)} new={len(todo)}: {",".join(todo[:8])}{"..." if len(todo) > 8 else ""}', flush=True)
            fresh = [p for p in todo if p not in state['backfilled']]
            if fresh:
                rc = run([PY, '-u', 'scripts/backfill_abstracts.py'] + [str(nbr / f'{p}.json') for p in fresh])
                if rc == 0:
                    state['backfilled'] += fresh
                    json.dump(state, open(state_f, 'w'), ensure_ascii=False, indent=1)
            ready = [p for p in todo if p in state['backfilled']]
            if ready:
                rc = run([PY, '-u', 'scripts/judge_batch.py', '--ext', str(ext), '--nbr', str(nbr),
                          '--out', str(outdir), '--pids', ','.join(ready)])
                if rc == 0:
                    state['judged'] += ready
                    json.dump(state, open(state_f, 'w'), ensure_ascii=False, indent=1)
        else:
            print(f'[driver] {time.strftime("%m-%d %H:%M")} closed={len(closed)}, nothing new', flush=True)

        if repair_errors(outdir, state):
            json.dump(state, open(state_f, 'w'), ensure_ascii=False, indent=1)
            run([PY, '-u', 'scripts/judge_batch.py', '--ext', str(ext), '--nbr', str(nbr),
                 '--out', str(outdir), '--pids', ','.join(sorted(set(state['judged'])))])

        if retrieval_done:
            print('[driver] retrieval ALL DONE -> final full backfill + judge', flush=True)
            run([PY, '-u', 'scripts/backfill_abstracts.py', str(nbr)])
            run([PY, '-u', 'scripts/judge_batch.py', '--ext', str(ext), '--nbr', str(nbr), '--out', str(outdir)])
            repair_errors(outdir, state)
            json.dump(state, open(state_f, 'w'), ensure_ascii=False, indent=1)
            run([PY, '-u', 'scripts/judge_batch.py', '--ext', str(ext), '--nbr', str(nbr), '--out', str(outdir), '--status'])
            print('DRIVER-DONE', flush=True)
            return
        time.sleep(a.interval)


if __name__ == '__main__':
    main()
