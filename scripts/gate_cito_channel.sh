#!/bin/bash
# cito 私有混合检索通道终测闸门(对照 gate_local_channel.sh)。
# 流程:FA0001-07 用 --channel cito --oa off 全流程重检(facet 配额+bge-m3 重排+pin)→
#       gpt-4o 在题率扫描 → 与 v6-API 基线(neighbor_relevance_v6.json)同批对比 + FA0007 seminal 核验。
# 及格线:ontopic 均值 ≥ v6-API 的 90%;seminal 必须命中。cito 单通道(OA off)对两通道 API 基线。
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=data/neighbors_gate_cito
mkdir -p $OUT
cp data/neighbors_s2/_queries.json $OUT/_queries.json

for pid in FA0001 FA0002 FA0003 FA0004 FA0005 FA0006 FA0007; do
  python3 scripts/retrieve_s2_slow.py --pid $pid --facet-alloc \
      --channel cito --oa off --out-dir $OUT
done

python3 scripts/scan_neighbor_relevance.py --nbr $OUT --out data/neighbor_relevance_gate_cito.json

python3 - << 'EOF'
import json, statistics, sys
ci = json.load(open('data/neighbor_relevance_gate_cito.json'))
v6 = json.load(open('data/neighbor_relevance_v6.json'))
keys = sorted(set(ci) & set(v6))
mc = statistics.mean(ci[k]['ontopic_count'] for k in keys if ci[k]['ontopic_count'] >= 0)
m6 = statistics.mean(v6[k]['ontopic_count'] for k in keys if v6[k]['ontopic_count'] >= 0)
bad_c = sum(1 for k in keys if ci[k]['ontopic_count'] <= 1)
bad_6 = sum(1 for k in keys if v6[k]['ontopic_count'] <= 1)
d = json.load(open('data/neighbors_gate_cito/FA0007.json'))
c1 = next(c for c in d['contributions'] if c['id'] == 'C1')
seminal = any('good word attack' in h['title'].lower() for h in c1['neighbors'])
print(f'同 {len(keys)} 条:cito 均值 {mc:.2f} vs v6-API {m6:.2f} ({100*mc/m6:.0f}%);bad {bad_c} vs {bad_6};seminal={seminal}')
ok = mc >= 0.9 * m6 and seminal
print('GATE-PASS' if ok else 'GATE-FAIL')
EOF
echo "CITO-GATE-DONE"
