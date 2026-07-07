#!/usr/bin/env python3
"""A4S 收尾:去重 → clean 子集 → 按 decision 分层统计。

步骤:
1. 全量 PDF md5 扫描,找出所有精确重复(不只依赖已知 7 对)。
2. clean 子集规则(原始 315 全保留,只是统计口径):
   - 精确 md5 重复:每组保留 pid 最小的一份,其余剔除;
   - 测试提交(OpenReview 标题 test/test2):剔除(AS0005 无实内容;AS0002 是 AS0004 的 md5 重复,已被上一条覆盖);
   - 元研究造假主题(AS0284 BadScientist / AS0328 注入审稿):从「诚信统计」剔除,不计入红旗率(其造数是研究设计);
   - 换壳/近重复(AS0201↔0202、AS0068↔0247)与同源簇:保留但打 flag,供敏感性分析。
3. 在 clean 子集上重跑分层统计(与 integrity 文档中全量口径同一套关键词),对比结论是否稳。

产物:data/agents4science/clean_subset.json(pid 列表 + 剔除原因 + flags)。
用法:python3 scripts/a4s_clean_stats.py
"""
import hashlib, json, os, pathlib, re
from collections import defaultdict

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT_DIR = pathlib.Path('data/extractions_a4s')
PDF_DIR = pathlib.Path('data/agents4science/pdfs')
OUT = pathlib.Path('data/agents4science/clean_subset.json')

META_FABRICATION = {'AS0284', 'AS0328'}   # 造假是研究主题,不是自身不可信
TEST_FORUMS = {'IfxYFkJEHx', '2p1CCmRoyr'}  # OpenReview 标题 test / test2
# 字节级近重复(PDF 大小差 <70B,同文重传;md5 不同但内容≡):剔除后者
CONTENT_DUPS = {'AS0114': 'AS0113', 'AS0339': 'AS0338'}
FLAG_PAIRS = [('AS0201', 'AS0202', 'rebrand'), ('AS0068', 'AS0247', 'near-dup'),
              ('AS0329', 'AS0331', 'trilogy'), ('AS0329', 'AS0332', 'trilogy')]

# --- 与 integrity 文档同口径的 notes 关键词 ---
RE_NEG = re.compile(r'negative result|null result|honest|fail(ed|ure)|no improvement|did not|worse than|underperform|negative finding|trade-?off|limitation', re.I)
RE_FAB = re.compile(r'np\.random|simulated|fabricat|made.?up|invented (data|numbers)|synthetic (results|numbers)|placeholder (data|results)|mock data', re.I)
RE_FLAG = re.compile(r'red flag|suspicious|inconsist|contradict|impossible|fabricat|np\.random|simulated|hallucinat|phantom|missing (section|appendix|table)|placeholder|circular|self-cit|not anonym|deanonymi|template', re.I)


def load_all():
    papers = {}
    for f in sorted(EXT_DIR.glob('AS*.json')):
        e = json.load(open(f))
        papers[e['paper_id']] = e
    return papers


def md5_groups(papers):
    by_md5 = defaultdict(list)
    for pid, e in papers.items():
        p = PDF_DIR / f"{e['forum']}.pdf"
        if p.exists():
            by_md5[hashlib.md5(p.read_bytes()).hexdigest()].append(pid)
    return {h: sorted(v) for h, v in by_md5.items() if len(v) > 1}


def stats(papers, pids, label):
    rows = defaultdict(lambda: dict(n=0, flag=0, neg=0, fab=0, prov=0))
    for pid in pids:
        e = papers[pid]
        d = e.get('decision') or 'no-decision'
        if d not in ('accept', 'reject', 'desk-reject'):
            d = 'other'
        notes = e.get('notes') or ''
        r = rows[d]
        r['n'] += 1
        r['flag'] += bool(RE_FLAG.search(notes))
        r['neg'] += bool(RE_NEG.search(notes))
        r['fab'] += bool(RE_FAB.search(notes))
        r['prov'] += bool(e.get('self_declared_provenance'))
    print(f'\n== {label} (n={len(pids)}) ==')
    print(f"{'decision':<12}{'n':>5}{'红旗%':>8}{'负结果%':>9}{'疑造数%':>9}{'prov%':>8}")
    for d in ('accept', 'reject', 'desk-reject', 'other'):
        if d not in rows:
            continue
        r = rows[d]
        pct = lambda k: f"{100*r[k]/r['n']:.0f}%"
        print(f"{d:<12}{r['n']:>5}{pct('flag'):>8}{pct('neg'):>9}{pct('fab'):>9}{pct('prov'):>8}")
    return rows


if __name__ == '__main__':
    papers = load_all()
    all_pids = sorted(papers)
    print(f'total extractions: {len(all_pids)}')

    dupes = md5_groups(papers)
    print(f'\nmd5 duplicate groups: {len(dupes)}')
    removed, reasons = set(), {}
    for h, grp in sorted(dupes.items(), key=lambda kv: kv[1]):
        keep, drop = grp[0], grp[1:]
        print(f'  {" ≡ ".join(grp)}  → keep {keep}')
        for pid in drop:
            removed.add(pid)
            reasons[pid] = f'md5-dup-of-{keep}'
    for pid, e in papers.items():
        if e['forum'] in TEST_FORUMS and pid not in removed:
            removed.add(pid)
            reasons[pid] = 'test-submission'
    for pid, keep in CONTENT_DUPS.items():
        if pid not in removed:
            removed.add(pid)
            reasons[pid] = f'content-dup-of-{keep}'

    clean = [p for p in all_pids if p not in removed]
    clean_integrity = [p for p in clean if p not in META_FABRICATION]

    flags = defaultdict(list)
    for a, b, why in FLAG_PAIRS:
        flags[a].append(f'{why}:{b}')
        flags[b].append(f'{why}:{a}')

    json.dump({
        'date': '2026-07-06',
        'total': len(all_pids),
        'removed': [{'pid': p, 'reason': reasons[p]} for p in sorted(removed)],
        'meta_fabrication_excluded_from_integrity': sorted(META_FABRICATION),
        'clean': clean,
        'clean_integrity': clean_integrity,
        'flags': dict(flags),
    }, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(f'\nclean: {len(clean)} | clean_integrity: {len(clean_integrity)} → {OUT}')

    stats(papers, all_pids, '全量 315(原口径,对照)')
    stats(papers, clean_integrity, 'clean 子集(去重+剔测试+剔元研究)')
