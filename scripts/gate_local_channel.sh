#!/bin/bash
# 纯本地通道终测闸门(OA 本地索引就绪后跑,一条命令出裁决)。
# 流程:FA0001-06(20 条验证贡献)用 --channel local --oa local 全流程重检 →
#       gpt-4o 在题率扫描 → 与 v6-API 基线同批对比 + FA0007 seminal 核验。
# 及格线(P1 验收标准的移植):ontopic 均值 ≥ v6-API 的 90%;seminal 必须命中。
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=data/neighbors_gate_val
mkdir -p $OUT
cp data/neighbors_s2/_queries.json $OUT/_queries.json

for pid in FA0001 FA0002 FA0003 FA0004 FA0005 FA0006 FA0007; do
  python3 scripts/retrieve_s2_slow.py --pid $pid --facet-alloc \
      --channel local --oa local --out-dir $OUT
done

python3 scripts/scan_neighbor_relevance.py --nbr $OUT --out data/neighbor_relevance_gate.json

python3 - << 'EOF'
import json, statistics, sys
ga = json.load(open('data/neighbor_relevance_gate.json'))
v6 = json.load(open('data/neighbor_relevance_v6.json'))
keys = sorted(set(ga) & set(v6))
mg = statistics.mean(ga[k]['ontopic_count'] for k in keys if ga[k]['ontopic_count'] >= 0)
m6 = statistics.mean(v6[k]['ontopic_count'] for k in keys if v6[k]['ontopic_count'] >= 0)
bad_g = sum(1 for k in keys if ga[k]['ontopic_count'] <= 1)
bad_6 = sum(1 for k in keys if v6[k]['ontopic_count'] <= 1)
d = json.load(open('data/neighbors_gate_val/FA0007.json'))
c1 = next(c for c in d['contributions'] if c['id'] == 'C1')
seminal = any('good word attack' in h['title'].lower() for h in c1['neighbors'])
print(f'同 {len(keys)} 条:local 均值 {mg:.2f} vs v6-API {m6:.2f} ({100*mg/m6:.0f}%);bad {bad_g} vs {bad_6};seminal={seminal}')
ok = mg >= 0.9 * m6 and seminal
print('GATE-PASS' if ok else 'GATE-FAIL')
sys.exit(0 if ok else 1)
EOF
