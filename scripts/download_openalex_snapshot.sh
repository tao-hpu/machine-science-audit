#!/bin/bash
# OpenAlex 官方快照下载(只取 works,实测 2026-06 快照 666GB;authors/sources 等小表用 API 够)。
# ⚠️ 2026-07-07:因本机仅 683GB 可用、与 S2 快照冲突,Tao 决定暂缓——等外置盘再跑本脚本。
# aws s3 sync 天然断点续传:已完成文件跳过,中断的文件下次整只重下。
# 公开桶免签名。合盖睡眠断网后 sync 会报错退出 → until 自愈循环 60s 后重试。
#
# 用法:
#   bash scripts/download_openalex_snapshot.sh            # 下载(可反复重跑)
#   bash scripts/download_openalex_snapshot.sh --status   # 看进度
set -u
cd "$(dirname "$0")/.."
DEST="${OPENALEX_DEST:-data/openalex_snapshot/works}"   # 外置盘时 OPENALEX_DEST=/Volumes/xxx/works 覆盖
mkdir -p "$DEST"

if [[ "${1:-}" == "--status" ]]; then
    echo "本地: $(du -sh "$DEST" 2>/dev/null | cut -f1), $(find "$DEST" -name '*.gz' | wc -l | tr -d ' ') 个 .gz 文件 (远端 2447 个 / 666GB)"
    exit 0
fi

# 新版桶结构(2026):data/jsonl/works/ 与 data/parquet/works/;旧的 data/works/ 已不存在
until aws s3 sync --no-sign-request --only-show-errors \
        "s3://openalex/data/jsonl/works/" "$DEST/"; do
    echo "[openalex] sync 中断 (rc=$?),60s 后重试 $(date '+%H:%M')" >&2
    sleep 60
done
echo "OPENALEX-SNAPSHOT-DONE"
