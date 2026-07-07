#!/usr/bin/env python3
"""G-base-3 前半:下载 166 篇镜像人类论文 PDF(OpenReview API + .or_token,裸 URL 403)。

下到 data/human_iclr2025/pdfs/<id>.pdf。断点续跑:已存在且 >10KB 的跳过。
不碰 S2,可与 v5 并行。用法:
    python3 scripts/download_human_pdfs.py            # 下载
    python3 scripts/download_human_pdfs.py --status   # 看进度
"""
import json, os, pathlib, sys, time
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
import openreview

MATCHED = pathlib.Path('data/human_iclr2025/matched_166.json')
PDF_DIR = pathlib.Path('data/human_iclr2025/pdfs')
TOKEN_CACHE = pathlib.Path('.or_token')
load_dotenv()

def get_client():
    if TOKEN_CACHE.exists():
        tok = TOKEN_CACHE.read_text().strip()
        try:
            c = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net', token=tok)
            c.get_group('venues'); return c
        except Exception:
            pass
    c = openreview.api.OpenReviewClient(
        baseurl='https://api2.openreview.net',
        username=os.environ['OPENREVIEW_USERNAME'],
        password=os.environ['OPENREVIEW_PASSWORD'])
    TOKEN_CACHE.write_text(c.token); return c

def done(pid):
    f = PDF_DIR / f'{pid}.pdf'
    return f.exists() and f.stat().st_size > 10240

if __name__ == '__main__':
    papers = json.load(open(MATCHED))
    if '--status' in sys.argv:
        print(f'pdfs: {sum(1 for p in papers if done(p["id"]))}/{len(papers)}'); sys.exit(0)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    client = get_client()
    fail = 0
    for i, p in enumerate(papers, 1):
        if done(p['id']):
            continue
        try:
            data = client.get_attachment('pdf', id=p['id'])
            if data[:4] != b'%PDF':
                print(f'[{i}] {p["id"]} NOT-PDF ({len(data)}B)', flush=True); fail += 1
            else:
                (PDF_DIR / f'{p["id"]}.pdf').write_bytes(data)
        except Exception as e:
            print(f'[{i}] {p["id"]} ERR {type(e).__name__}: {str(e)[:80]}', flush=True)
            fail += 1; time.sleep(10)
        if i % 25 == 0:
            print(f'  ...{i}/{len(papers)} (fail {fail})', flush=True)
        time.sleep(1.5)
    print(f'PDF-DONE {sum(1 for p in papers if done(p["id"]))}/{len(papers)} (fail {fail})', flush=True)
