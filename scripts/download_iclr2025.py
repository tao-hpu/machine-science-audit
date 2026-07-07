"""G-base-1:抓 ICLR 2025 人类论文作机器语料的对照基线池。

只碰 OpenReview,不碰 S2,可与 v5 检索重跑并行。
拉全部 submissions 的元数据(title/abstract/keywords/venue/venueid/pdf/cdate),
按 venueid/venue 分桶(accept/reject/withdrawn/desk),accept+reject 作后续主题匹配抽样池。
不拉 reviews(G-base-1 只要名单+元数据;RQ4 要评审分再单独拉)。

凭据从 .env 读,复用 .or_token 缓存,绝不打印。用法:
    python3 scripts/download_iclr2025.py --probe    # 先抓 25 篇看 venue 分桶口径
    python3 scripts/download_iclr2025.py            # 全量拉元数据
    python3 scripts/download_iclr2025.py --status    # 看已抓进度
"""
import os, sys, json, pathlib
from collections import Counter
from dotenv import load_dotenv
import openreview

VENUE = "ICLR.cc/2025/Conference"
OUT = pathlib.Path("data/human_iclr2025")
TOKEN_CACHE = pathlib.Path(".or_token")  # gitignored

load_dotenv()


def get_client():
    """优先用缓存 token,失效再登录一次(避免 429 登录限流)。"""
    if TOKEN_CACHE.exists():
        tok = TOKEN_CACHE.read_text().strip()
        try:
            c = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net", token=tok)
            c.get_group("venues")  # 探活
            return c
        except Exception:
            pass
    c = openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=os.environ["OPENREVIEW_USERNAME"],
        password=os.environ["OPENREVIEW_PASSWORD"],
    )
    TOKEN_CACHE.write_text(c.token)
    return c


def flat(content):
    """API2 content 是 {k: {'value': v}},拍平成 {k: v}。"""
    return {k: (v.get("value") if isinstance(v, dict) else v) for k, v in content.items()}


def bucket(venueid, venue):
    """按 venueid/venue 归 accept/reject/withdrawn/desk/other。"""
    vid = (venueid or "").lower()
    ven = (venue or "").lower()
    if "withdraw" in vid or "withdraw" in ven:
        return "withdrawn"
    if "desk" in vid or "desk" in ven:
        return "desk"
    if "reject" in vid or "reject" in ven:
        return "reject"
    # 录用:venueid 就是主 venue 且 venue 写着 poster/oral/spotlight
    if vid == VENUE.lower() or any(t in ven for t in ("poster", "oral", "spotlight")):
        return "accept"
    if "submitted to" in ven:  # 有决定前/未决 = 视作 reject 池外,单列
        return "reject"
    return "other"


def record(s):
    c = flat(s.content)
    return {
        "id": s.id, "forum": s.forum, "number": s.number, "cdate": s.cdate,
        "title": c.get("title", ""),
        "abstract": c.get("abstract", ""),
        "keywords": c.get("keywords", []),
        "venue": c.get("venue", ""),
        "venueid": c.get("venueid", ""),
        "pdf": c.get("pdf", ""),  # 相对路径 /pdf/xxx.pdf,G-base-3 下载用
        "decision": bucket(c.get("venueid", ""), c.get("venue", "")),
    }


def probe():
    c = get_client()
    notes = c.get_notes(invitation=f"{VENUE}/-/Submission", limit=25)
    print(f"[probe] 抓到 {len(notes)} 篇")
    recs = [record(n) for n in notes]
    print("venueid 分布:", dict(Counter(r["venueid"] for r in recs)))
    print("venue 分布:", dict(Counter(r["venue"] for r in recs)))
    print("decision 分桶:", dict(Counter(r["decision"] for r in recs)))
    print("\n样例 3 条:")
    for r in recs[:3]:
        print(f"  #{r['number']} [{r['decision']}] {r['title'][:70]}")
        print(f"     venue='{r['venue']}' venueid='{r['venueid']}' pdf={'Y' if r['pdf'] else 'N'} kw={len(r['keywords'])}")


def status():
    f = OUT / "submissions.json"
    if not f.exists():
        print("未抓取"); return
    recs = json.loads(f.read_text())
    print(f"总计 {len(recs)} 篇")
    print("decision 分桶:", dict(Counter(r["decision"] for r in recs)))


def main():
    if "--probe" in sys.argv:
        probe(); return
    if "--status" in sys.argv:
        status(); return
    OUT.mkdir(parents=True, exist_ok=True)
    c = get_client()
    print("[fetch] 拉全部 submissions 元数据(约 1.1 万篇,分页,请稍候)...")
    notes = c.get_all_notes(invitation=f"{VENUE}/-/Submission")
    print(f"[fetch] {len(notes)} 篇")
    recs = [record(n) for n in notes]
    (OUT / "submissions.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1))
    dist = Counter(r["decision"] for r in recs)
    print(f"[done] {len(recs)} -> {OUT/'submissions.json'}")
    print("decision 分桶:", dict(dist))
    # accept+reject = 主题匹配抽样池
    pool = [r for r in recs if r["decision"] in ("accept", "reject")]
    (OUT / "pool_accept_reject.json").write_text(json.dumps(pool, ensure_ascii=False, indent=1))
    print(f"[pool] accept+reject 抽样池 {len(pool)} 篇 -> pool_accept_reject.json")


if __name__ == "__main__":
    main()
