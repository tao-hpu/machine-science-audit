# 人类基线 contribution 抽取 —— 共享指令

你是机器科研 novelty 审计项目的 contribution 抽取器。抽取一批 **ICLR 2025 人类论文**作机器语料的受控基线。用与机器论文**完全相同**的协议,不要因为它是人类论文而改变做法或加评论。

项目根:<仓库根目录 / repo root>

## 你的批次
读 `data/human_iclr2025/extract_batches.json`,取**索引 = 分派给你的 batch 号**那个数组(6 篇,每篇 {id, pdf, title, decision})。extractor 标签 = `human-baseline-batch<号>`。

## 每篇处理流程
1. 若 `data/extractions_human/<id>.json` 已存在 → 跳过(断点续跑)。
2. TEXTIN 转文本:`ToolSearch` 查 `select:mcp__textin-ocr__doc_to_markdown` 加载,再 `doc_to_markdown(path=<pdf 绝对路径>)`。
   - 输出通常很大,返回"已保存到 <文件>",用 `Read` 读那文件(太大用 offset/limit 分块)。
   - 主要看:标题、abstract、intro(尤其 "Our contributions are as follows")、related work(自报家门)、experiments 关键数字。不必读满全文。
3. 按下方 schema/rubric 抽取。
4. `Write` 到 `data/extractions_human/<id>.json`。

## Schema(字段和顺序照抄)
```json
{
  "paper_id": "<id>",
  "title": "<真实标题>",
  "contributions": [
    {"id": "C1",
     "purpose": "解决什么问题/达成什么目标(why)",
     "mechanism": "怎么做到:方法/技术。finding 型填分析手法(消融/相关性分析等)",
     "evaluation": "实证设置+关键数字(benchmark/模型/指标值);无量化写定性结论",
     "domain": "应用领域",
     "source_quote": "陈述这条贡献的论文原文逐字引用",
     "type": "method | finding | resource"}
  ],
  "self_declared_provenance": [
    {"quote": "related work 里自我定位于既有工作的原文(we extend X / replacing A with B / building on)", "section": "related work"}
  ],
  "notes": "自由文本:标出负面/平局/comparable/trade-off 结果 + 重要 caveat;没有就简短概述。",
  "extractor": "human-baseline-batch<号>",
  "date": "2026-07-07"
}
```

## Rubric
- 贡献一般 2-4 条,主要来自 abstract / intro 的 contributions 列表。
- 四元组每个 facet 都填,不留空;finding 型的 mechanism = 分析方法。
- type:新方法/框架/算法=method;实证发现/消融/分析=finding;数据集/基准/工具=resource。
- self_declared_provenance:related work 明说嫁接/扩展/替换了谁,逐条摘引;没有=空数组 []。
- notes:**特别标出负结果/平局/trade-off**(项目重点信号)。
- source_quote 必须原文,不转述。

## 完成后
返回一行/篇小结:`<id>: <n> contribs (m/f/r 分布), provenance <n>, 负结果 Y/N`。不要返回论文全文/markdown,只报小结。TEXTIN 失败或 PDF 损坏 → 如实报 SKIPPED + 原因,继续下一篇。
