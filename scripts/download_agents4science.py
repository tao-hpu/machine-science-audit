"""Agents4Science/2025/Conference 语料抓取。

登录一次(token 缓存到 .or_token,复用避免 429 登录限流),
拉全部 submissions + 每篇 forum 上的评审/元评审等 replies,
存到 data/agents4science/{submissions.json, reviews/<forum>.json, manifest.json}。

凭据从 .env 读,绝不打印。用法:
    python3 scripts/download_agents4science.py            # 抓取
    python3 scripts/download_agents4science.py --status   # 只看已抓进度
"""
import os, sys, json, time, pathlib
from dotenv import load_dotenv
import openreview

VENUE = "Agents4Science/2025/Conference"
OUT = pathlib.Path("data/agents4science")
TOKEN_CACHE = pathlib.Path(".or_token")  # gitignored

load_dotenv()


def get_client():
    """优先用缓存 token,失效再登录一次。"""
    if TOKEN_CACHE.exists():
        tok = TOKEN_CACHE.read_text().strip()
        try:
            c = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net", token=tok)
            c.get_group("venues")  # 探活
            return c
        except Exception:
            pass  # token 过期,重登
    c = openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=os.environ["OPENREVIEW_USERNAME"],
        password=os.environ["OPENREVIEW_PASSWORD"],
    )
    TOKEN_CACHE.write_text(c.token)
    return c


def status():
    subf = OUT / "submissions.json"
    n_sub = len(json.loads(subf.read_text())) if subf.exists() else 0
    n_rev = len(list((OUT / "reviews").glob("*.json"))) if (OUT / "reviews").exists() else 0
    print(f"submissions saved: {n_sub}")
    print(f"forums with replies saved: {n_rev}")


def main():
    if "--status" in sys.argv:
        status()
        return
    (OUT / "reviews").mkdir(parents=True, exist_ok=True)
    c = get_client()

    subs = c.get_all_notes(invitation=f"{VENUE}/-/Submission", details="replies")
    print(f"[submissions] {len(subs)}")

    sub_records, manifest = [], []
    for i, s in enumerate(subs, 1):
        content = {k: v.get("value") if isinstance(v, dict) else v for k, v in s.content.items()}
        sub_records.append({"id": s.id, "forum": s.forum, "number": s.number,
                            "cdate": s.cdate, "content": content})
        # replies 已随 details=replies 返回,分类落盘
        replies = s.details.get("replies", []) if s.details else []
        (OUT / "reviews" / f"{s.forum}.json").write_text(json.dumps(replies, ensure_ascii=False, indent=2))
        inv_types = sorted({(r.get("invitations") or ["?"])[0].split("/-/")[-1] for r in replies})
        manifest.append({"forum": s.forum, "number": s.number,
                        "title": content.get("title", "")[:120],
                        "n_replies": len(replies), "reply_types": inv_types})
        if i % 10 == 0:
            print(f"  ...{i}/{len(subs)}")

    (OUT / "submissions.json").write_text(json.dumps(sub_records, ensure_ascii=False, indent=2))
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[done] {len(sub_records)} submissions -> {OUT}")
    # 汇总 reply 类型分布,方便定 novelty 分抽取口径
    from collections import Counter
    cnt = Counter(t for m in manifest for t in m["reply_types"])
    print("reply types across corpus:", dict(cnt))


if __name__ == "__main__":
    main()
