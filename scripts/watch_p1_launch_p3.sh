#!/bin/bash
# P1(v6 机器重选)ALL DONE 后自动接力 P3(人类检索)+ P4 流水线驱动。
# P5(A4S)不自动启动——有 sanity 闸门,等 P2+P4 出数、人工过形状后再点火。
cd "$(dirname "$0")/.." || exit 1
until grep -q "ALL DONE" logs/retrieve_v6.log 2>/dev/null; do sleep 300; done
echo "[watcher] P1 ALL DONE detected at $(date '+%m-%d %H:%M'), launching P3 + P4 driver"
S2_MIN_INTERVAL=3 nohup python3 -u scripts/retrieve_s2_slow.py \
    --extract-dir data/extractions_human --out-dir data/neighbors_human \
    --facet-alloc --cutoff-source human --k 20 >> logs/retrieve_human.log 2>&1 &
echo "[watcher] P3 retrieval PID $!"
sleep 60
nohup python3 -u scripts/pipeline_driver.py \
    --ext data/extractions_human --nbr data/neighbors_human --out data/judgments_human \
    --retrieve-log logs/retrieve_human.log >> logs/pipeline_human.log 2>&1 &
echo "[watcher] P4 driver PID $!"
