"""G-base-2:可复现主题匹配 —— 从 8613 篇 ICLR 2025 人类论文里,为每篇 FARS 机器论文
匹配一篇主题最近的人类论文(贪心 1:1 无放回),得到 166 篇镜像 FARS 主题分布的人类基线子集。

协议(写进论文 Methods,可复现):
  1. 两边统一表示 = title + abstract。机器侧 abstract 从 FARS PDF 抽(pypdf,缓存)。
  2. bge-m3 embed(与检索层同款 embedding),cosine 相似度。
  3. 贪心 1:1 无放回:按「最难匹配者优先」(best-match 相似度升序)处理机器论文,
     每篇选当前未被占用的最近人类论文。避免先到先得把好邻居抢光。
  4. 报告匹配距离分布 + 自然落出的 accept/reject 比,供质量核查。

只碰 bge-m3 embed 机(另一台),不碰 S2,可与 v5 并行。用法:
    python3 scripts/match_human_baseline.py            # 跑匹配
    python3 scripts/match_human_baseline.py --report   # 只看已产出子集的报告
"""
import os, sys, json, math, re, pathlib
from collections import Counter
import urllib.request
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXTRACT_DIR = ROOT / "data" / "extractions"
FARS_PDF = ROOT / "data" / "fars-a-reviews" / "data"
HUMAN_POOL = ROOT / "data" / "human_iclr2025" / "pool_accept_reject.json"
OUT_DIR = ROOT / "data" / "human_iclr2025"
FARS_ABS = OUT_DIR / "fars_abstracts.json"          # 机器侧 abstract 缓存
EMB_CACHE = OUT_DIR / "_embcache.json"              # embedding 缓存(按文本 hash)
MATCHED = OUT_DIR / "matched_166.json"

load_dotenv()
EMBED_BASE = os.environ["EMBED_API_BASE"]
EMBED_MODEL = os.environ["EMBED_MODEL"]


def embed_batch(texts):
    req = urllib.request.Request(
        f"{EMBED_BASE}/embeddings",
        data=json.dumps({"model": EMBED_MODEL, "input": texts}).encode(),
        headers={"Authorization": "Bearer NO_NEED", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return [d["embedding"] for d in json.load(r)["data"]]


def embed_all(texts, cache):
    """按文本缓存 embedding,只算没见过的,分批 64。"""
    todo = [t for t in texts if t not in cache]
    for i in range(0, len(todo), 64):
        chunk = todo[i:i + 64]
        for t, v in zip(chunk, embed_batch(chunk)):
            cache[t] = v
        print(f"    embed {min(i+64,len(todo))}/{len(todo)}", flush=True)
    return [cache[t] for t in texts]


def cos(a, b):
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return s / (na * nb) if na and nb else 0.0


def fars_abstract(pid):
    import pypdf
    r = pypdf.PdfReader(str(FARS_PDF / pid / f"{pid}.pdf"))
    txt = "\n".join(p.extract_text() or "" for p in r.pages[:2])
    m = re.search(r"abstract(.{50,2500}?)(\n\s*1\s|\nintroduction|\n1[\.\s]+introduction)",
                  txt, re.I | re.S)
    return re.sub(r"\s+", " ", (m.group(1) if m else "")).strip()


def load_fars():
    """166 篇机器论文:title(抽取文件)+ abstract(PDF,缓存)。"""
    abs_cache = json.loads(FARS_ABS.read_text()) if FARS_ABS.exists() else {}
    papers = []
    for f in sorted(EXTRACT_DIR.glob("FA*.json")):
        d = json.loads(f.read_text()); pid = d["paper_id"]
        if pid not in abs_cache:
            try:
                abs_cache[pid] = fars_abstract(pid)
            except Exception as e:
                print(f"    !! {pid} abstract fail: {type(e).__name__}"); abs_cache[pid] = ""
        papers.append({"pid": pid, "title": d.get("title", ""), "abstract": abs_cache[pid]})
    FARS_ABS.write_text(json.dumps(abs_cache, ensure_ascii=False, indent=1))
    return papers


def rep(title, abstract):
    return f"{title}. {(abstract or '')[:1500]}"


def report(matched):
    sims = sorted(m["sim"] for m in matched)
    n = len(sims)
    q = lambda p: sims[int(p * (n - 1))]
    print(f"\n[report] 匹配 {n} 篇人类论文")
    print(f"  sim 分布: min={sims[0]:.3f} p25={q(.25):.3f} median={q(.5):.3f} p75={q(.75):.3f} max={sims[-1]:.3f}")
    print(f"  弱匹配(sim<0.5): {sum(1 for s in sims if s<0.5)} 篇  <0.4: {sum(1 for s in sims if s<0.4)} 篇")
    print(f"  自然落出的决定分布: {dict(Counter(m['decision'] for m in matched))}")


def main():
    if "--report" in sys.argv:
        report(json.loads(MATCHED.read_text())); return

    fars = load_fars()
    pool = json.loads(HUMAN_POOL.read_text())
    print(f"[load] FARS {len(fars)} 篇, 人类池 {len(pool)} 篇")

    cache = json.loads(EMB_CACHE.read_text()) if EMB_CACHE.exists() else {}
    print("[embed] 机器侧...")
    fv = embed_all([rep(p["title"], p["abstract"]) for p in fars], cache)
    print("[embed] 人类侧(8613,首次约几分钟)...")
    hv = embed_all([rep(p["title"], p["abstract"]) for p in pool], cache)
    EMB_CACHE.write_text(json.dumps(cache))

    # 每篇机器论文对全人类池的相似度,取 best 供排序
    print("[match] 计算相似度 + 贪心 1:1 无放回...")
    best = []
    for i, v in enumerate(fv):
        sims = [(cos(v, hv[j]), j) for j in range(len(pool))]
        sims.sort(reverse=True)
        best.append((sims[0][0], i, sims))  # (best_sim, machine_idx, 排序候选)
    # 最难匹配者优先(best_sim 升序),给它先挑
    best.sort(key=lambda x: x[0])
    used, matched = set(), []
    for _, i, sims in best:
        for s, j in sims:
            if j not in used:
                used.add(j)
                h = pool[j]
                matched.append({**h, "matched_to": fars[i]["pid"],
                                "matched_title": fars[i]["title"], "sim": round(s, 4)})
                break
    matched.sort(key=lambda m: m["matched_to"])
    MATCHED.write_text(json.dumps(matched, ensure_ascii=False, indent=1))
    print(f"[done] {len(matched)} 篇 -> {MATCHED}")
    report(matched)


if __name__ == "__main__":
    main()
