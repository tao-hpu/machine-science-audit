#!/usr/bin/env python3
"""Agents4Science 315 篇 PDF 下载。

裸 URL 403(OpenReview PDF 需登录),走 API 客户端 + .or_token 缓存
(复用 download_agents4science.py 的 token,登录每窗口限 3 次,绝不反复登)。
下到 data/agents4science/pdfs/<forum>.pdf。断点续跑:已存在且 >10KB 的跳过。

用法:
    python3 scripts/download_a4s_pdfs.py            # 下载
    python3 scripts/download_a4s_pdfs.py --status   # 看进度
"""
import json, os, pathlib, sys, time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
import openreview

SUBS = pathlib.Path('data/agents4science/submissions.json')
PDF_DIR = pathlib.Path('data/agents4science/pdfs')
TOKEN_CACHE = pathlib.Path('.or_token')
load_dotenv()

def get_client():
    if TOKEN_CACHE.exists():
        tok = TOKEN_CACHE.read_text().strip()
        try:
            c = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net', token=tok)
            c.get_group('venues')
            return c
        except Exception:
            pass
    c = openreview.api.OpenReviewClient(
        baseurl='https://api2.openreview.net',
        username=os.environ['OPENREVIEW_USERNAME'],
        password=os.environ['OPENREVIEW_PASSWORD'])
    TOKEN_CACHE.write_text(c.token)
    return c

def done(forum):
    f = PDF_DIR / f'{forum}.pdf'
    return f.exists() and f.stat().st_size > 10240

if __name__ == '__main__':
    subs = [s for s in json.load(open(SUBS)) if s['content'].get('pdf')]
    if '--status' in sys.argv:
        print(f'pdfs: {sum(1 for s in subs if done(s["forum"]))}/{len(subs)}')
        sys.exit(0)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    client = get_client()
    fail = 0
    for i, s in enumerate(subs, 1):
        if done(s['forum']):
            continue
        try:
            data = client.get_attachment('pdf', id=s['id'])
            if data[:4] != b'%PDF':
                print(f'[{i}] {s["forum"]} NOT-PDF ({len(data)}B)', flush=True)
                fail += 1
            else:
                (PDF_DIR / f'{s["forum"]}.pdf').write_bytes(data)
        except Exception as e:
            print(f'[{i}] {s["forum"]} ERR {type(e).__name__}: {str(e)[:80]}', flush=True)
            fail += 1
            time.sleep(10)
        if i % 25 == 0:
            print(f'  ...{i}/{len(subs)} (fail {fail})', flush=True)
        time.sleep(1.5)
    print(f'PDF-DONE {sum(1 for s in subs if done(s["forum"]))}/{len(subs)} (fail {fail})', flush=True)
