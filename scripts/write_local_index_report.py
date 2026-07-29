#!/usr/bin/env python3
"""汇编 docs/_local-index-report.md(任务书「完成后」条款):build 统计 + validate 两个数。"""
import json, os, pathlib, sqlite3

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = pathlib.Path('data/s2_local_index')

stats = json.loads((OUT / '_build_stats.json').read_text())
val = json.loads(pathlib.Path('data/neighbors_local_val/_validation.json').read_text())
con = sqlite3.connect(OUT / 'abstracts.db')
kv_n = con.execute('SELECT COUNT(*) FROM kv').fetchone()[0]
con.close()
idx_gb = sum(f.stat().st_size for f in (OUT / 'tantivy').rglob('*')) / 1e9
db_gb = (OUT / 'abstracts.db').stat().st_size / 1e9

# --- OpenAlex 第二通道(索引在外置盘;未挂载/未建则跳过该段)---
OA_OUT = pathlib.Path(os.environ.get('OA_LOCAL_INDEX_DIR', "/Volumes/TONY'S WD/data/oa_local_index"))
oa_section = ''
oa_sf = OA_OUT / '_build_stats.json'
if oa_sf.exists():
    o = json.loads(oa_sf.read_text())
    o_docs = max(o['docs'], 1)
    o_total = o['docs'] + o['skip_type'] + o['skip_notitle']
    o_idx_gb = sum(f.stat().st_size for f in (OA_OUT / 'tantivy').rglob('*') if f.is_file()) / 1e9
    dt = sorted(o.get('dropped_types', {}).items(), key=lambda x: -x[1])[:6]
    drop_line = ', '.join(f'{k} {v:,}' for k, v in dt)
    oa_section = f"""
## OpenAlex 本地索引(第二通道,`--oa local`)

自动生成(build_oa_index.py)。索引在外置盘 WD(内置盘容不下),摘要内联无独立 KV。

- 语料:OpenAlex works 快照(2026-06 冻结,{o['files']} 个分片)
- tantivy 索引:**{o['docs']:,} docs**,体积 {o_idx_gb:.0f} GB
- 有摘要占比:**{o['with_abstract'] / o_docs:.1%}**({o['with_abstract']:,} 条 —— 补 S2 快照 19.4% 摘要缺口的主力)
- 无日期(date=0)占比:**{o['date0'] / o_docs:.1%}**(检索时排除防泄漏)
- type 过滤:保留论文类 {o['docs']:,} / 总记录 {o_total:,}(**{o['docs'] / o_total:.1%}**);丢弃非论文条目 top:{drop_line}
"""


report = f"""# 本地检索索引 —— 构建与验证报告

自动生成(write_local_index_report.py)。任务书:`docs/_local-index-task.md`。

## Build 统计

- 语料:S2 快照 release `2026-06-24`(冻结,可复现)
- tantivy 索引:**{stats['docs']:,} docs**({stats['files']} 个 papers 分片),体积 {idx_gb:.1f} GB
- 摘要 KV:{kv_n:,} 条(sqlite {db_gb:.1f} GB,截 800 字符)
- 有摘要占比:**{stats['with_abstract'] / max(stats['docs'], 1):.1%}**(论文附录用)
- 无日期(date=0)占比:**{stats['date0'] / max(stats['docs'], 1):.1%}**(检索时排除防泄漏,论文附录用)

## Validate(20 条验证批 + FA0007/C1,产数据不下结论)

- ① FA0007/C1 seminal 金标准(Lowd & Meek 2005)命中:**{val['gold_FA0007_C1_lowd_meek_2005']}**
- ② 对 neighbors_facet_val 已选近邻标题覆盖率:**{val['coverage_overall']}**(OpenAlex 参照 30%/13%)
  - 分篇:{'; '.join(val['coverage_per_paper'])}
- 过滤中因 date=0 排除的命中:{val['excluded_nodate_hits']}
- 候选池:`data/neighbors_local_val/`(未重排,在题率扫描由主线程做)
{oa_section}
## 主线程接手项

质量闸门(在题率扫描:tantivy-S2 主通道 + tantivy-OA 第二通道 vs API 基线,20 条验证批)→
过线则 retrieve_s2_slow 切 `--channel local --oa local`,三语料统一本地重跑。
"""
pathlib.Path('docs/_local-index-report.md').write_text(report)
print(report)
