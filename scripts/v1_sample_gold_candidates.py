#!/usr/bin/env python3
"""V1 gold-recall 抽样:两臂分层抽贡献,加手工种子。

用法: python3 scripts/v1_sample_gold_candidates.py
输出: data/v1_gold/sample.json

分层:每臂 facet-novel 10 / recombination 5 / covered 3(seed 42),
尽量一篇论文只抽一条贡献,摊薄单篇 PDF 的阅读成本。
机器臂另加 3 条 gold_neighbors 手工种子(FA0007/FA0077/FA0102,取该篇 C1 或
已有金标对应的贡献,只作 seed 标记,不占分层名额)。
"""
import json
import os
import random

ARMS = {
    'machine': ('data/judgments_fars_rescue', 'data/extractions'),
    'human': ('data/judgments_human_rescue', 'data/extractions_human'),
}
STRATA = {'facet-novel': 10, 'recombination': 5, 'covered': 3}
SEED_PAPERS = ['FA0007', 'FA0077', 'FA0102']  # data/gold_neighbors 手工金标

random.seed(42)


def load_rows(jdir):
    rows = []
    for fn in sorted(os.listdir(jdir)):
        if not fn.endswith('.json') or fn.startswith('_'):
            continue
        d = json.load(open(os.path.join(jdir, fn)))
        for c in d.get('contributions', []):
            state = c.get('consensus_state_v2') or c.get('consensus_state')
            rows.append({
                'paper_id': d['paper_id'],
                'title': d.get('title', ''),
                'cid': c['cid'],
                'state': state,
                'type': c.get('type'),
            })
    return rows


def stratified(rows, strata):
    picked, used_papers = [], set()
    for state, n in strata.items():
        pool = [r for r in rows if r['state'] == state]
        random.shuffle(pool)
        got = 0
        for r in pool:  # 先取未用过的论文
            if got >= n:
                break
            if r['paper_id'] in used_papers:
                continue
            picked.append(r)
            used_papers.add(r['paper_id'])
            got += 1
        for r in pool:  # 论文不够就放开重复
            if got >= n:
                break
            if r in picked:
                continue
            picked.append(r)
            got += 1
    return picked


def main():
    out = {'seed': 42, 'strata': STRATA, 'arms': {}}
    for arm, (jdir, _) in ARMS.items():
        rows = load_rows(jdir)
        picked = stratified(rows, STRATA)
        for r in picked:
            r['source'] = 'stratified'
        if arm == 'machine':
            for pid in SEED_PAPERS:
                d = json.load(open(os.path.join(jdir, f'{pid}.json')))
                c = d['contributions'][0]
                picked.append({
                    'paper_id': pid, 'title': d.get('title', ''),
                    'cid': c['cid'],
                    'state': c.get('consensus_state_v2') or c.get('consensus_state'),
                    'type': c.get('type'), 'source': 'hand-seed',
                })
        out['arms'][arm] = picked
        print(f'{arm}: {len(picked)} contributions, '
              f'{len(set(r["paper_id"] for r in picked))} papers')
    os.makedirs('data/v1_gold', exist_ok=True)
    with open('data/v1_gold/sample.json', 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('wrote data/v1_gold/sample.json')


if __name__ == '__main__':
    main()
