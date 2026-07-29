#!/bin/bash
# 三语料纯本地检索 + 判定流水线总编排(gate_local_channel.sh 打出 GATE-PASS 后跑)。
#
# 结构(检索本地极快,顺序跑防 embedding 服务过载;判定驱动跟随各语料):
#   FARS  : extract=data/extractions       → nbr=data/neighbors_fars_local → judg=data/judgments_fars
#   human : extract=data/extractions_human → nbr=data/neighbors_human      → judg=data/judgments_human
#   A4S   : extract=data/extractions_a4s   → nbr=data/neighbors_a4s        → judg=手动(sanity 闸门后,见尾注)
# 判定驱动只自动启 FARS+human;A4S 判定(~$45-70)等两侧 sanity 过了手动点。
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs data/neighbors_fars_local

# 查询缓存就位(FARS 缓存在 v5 目录,复制;human/a4s 预热时已在各自目录)
cp -n data/neighbors_s2/_queries.json data/neighbors_fars_local/_queries.json 2>/dev/null || true

run_retrieve () {  # $1=extract $2=outdir $3=cutoff-source $4=log
  nohup python3 -u scripts/retrieve_s2_slow.py --extract-dir "$1" --out-dir "$2" \
      --facet-alloc --cutoff-source "$3" --channel local --oa local --dump-pool --k 20 \
      >> "logs/$4" 2>&1
}

echo "[pipeline] FARS 检索启动 $(date '+%H:%M')"
run_retrieve data/extractions        data/neighbors_fars_local fars  retrieve_fars_local.log
echo "[pipeline] human 检索启动 $(date '+%H:%M')"
run_retrieve data/extractions_human  data/neighbors_human      human retrieve_human_local.log
echo "[pipeline] a4s 检索启动 $(date '+%H:%M')"
run_retrieve data/extractions_a4s    data/neighbors_a4s        a4s   retrieve_a4s_local.log

echo "[pipeline] 检索全部完成,启动 FARS+human 判定驱动 $(date '+%H:%M')"
nohup python3 -u scripts/pipeline_driver.py --ext data/extractions --nbr data/neighbors_fars_local \
    --out data/judgments_fars --retrieve-log logs/retrieve_fars_local.log --interval 600 \
    >> logs/pipeline_fars.log 2>&1 &
nohup python3 -u scripts/pipeline_driver.py --ext data/extractions_human --nbr data/neighbors_human \
    --out data/judgments_human --retrieve-log logs/retrieve_human_local.log --interval 600 \
    >> logs/pipeline_human.log 2>&1 &
echo "[pipeline] 驱动已挂后台。sanity 过线后手动启 A4S 判定:"
echo "  nohup python3 -u scripts/pipeline_driver.py --ext data/extractions_a4s --nbr data/neighbors_a4s --out data/judgments_a4s --retrieve-log logs/retrieve_a4s_local.log --interval 600 >> logs/pipeline_a4s.log 2>&1 &"
echo "  sanity 看数:python3 scripts/report_judgments.py data/judgments_fars data/judgments_human --decisions data/human_iclr2025/matched_166.json"
