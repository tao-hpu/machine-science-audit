#!/usr/bin/env python3
"""S2 官方离线数据集下载(papers + abstracts),断点续传。

设计要点:
- 元数据端点(拿预签名 S3 URL)与检索共享 key 的 1 req/s 限速 → 429 时耐心退避重试,
  趁 v6 检索请求间隙挤进去;真正下载走 S3,不占 key 限速。
- release 首次运行锁定到 data/s2_snapshot/_release.txt,续跑不换 release(文件集一致)。
- 每文件 curl -C - 续传,成功后落 <file>.done 标记;预签名 URL 过期(403)自动重取。
- 合盖睡眠会断 TCP:醒来后 curl 失败 → 外层循环重取 URL 继续,或直接重跑本脚本。

用法:
    python3 scripts/download_s2_snapshot.py             # 下载(可反复重跑续传)
    python3 scripts/download_s2_snapshot.py --status    # 看进度
"""
import json, os, pathlib, random, subprocess, sys, time, urllib.parse, urllib.request

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

API = 'https://api.semanticscholar.org/datasets/v1'
DATASETS = ['papers', 'abstracts']   # citations/embeddings 单独数百 GB,本机不下
ROOT = pathlib.Path(os.environ.get('S2_SNAPSHOT_DIR', 'data/s2_snapshot'))   # 可 .env 覆盖
KEY = os.environ['S2_API_KEY']


def api_get(path):
    """带 key 请求元数据端点。v6 检索进程占着 key 的 1 req/s → 429 常态;
    随机抖动 2-6s 无限重试,挤它两次请求之间的空闲秒。绝不因重试耗尽退出。"""
    url = f'{API}/{path}'
    n = 0
    while True:
        try:
            req = urllib.request.Request(url, headers={'x-api-key': KEY})
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.load(r)
            if 'code' not in d:   # 429 有时以 200+body 形式返回
                return d
        except Exception as e:
            n += 1
            if n % 30 == 0:   # 别刷屏,每 30 次报一行
                print(f'  [api] 已重试 {n} 次, 最近: {type(e).__name__} {str(e)[:50]}', flush=True)
        time.sleep(random.uniform(2, 6))


def get_release():
    f = ROOT / '_release.txt'
    if f.exists():
        return f.read_text().strip()
    rid = api_get('release/latest')['release_id']
    ROOT.mkdir(parents=True, exist_ok=True)
    f.write_text(rid)
    return rid


def fname_of(url):
    return pathlib.Path(urllib.parse.urlparse(url).path).name


def fetch_manifest(rid, ds):
    """拿(新鲜的)预签名 URL 列表;首次同时落 _manifest 记录文件名全集供 --status 用。"""
    d = api_get(f'release/{rid}/dataset/{ds}')
    files = d['files']
    mf = ROOT / ds / '_manifest.json'
    mf.parent.mkdir(parents=True, exist_ok=True)
    if not mf.exists():
        mf.write_text(json.dumps([fname_of(u) for u in files], indent=1))
    return files


def status():
    rid = (ROOT / '_release.txt').read_text().strip() if (ROOT / '_release.txt').exists() else '?'
    print(f'release: {rid}')
    for ds in DATASETS:
        mf = ROOT / ds / '_manifest.json'
        if not mf.exists():
            print(f'{ds}: 未开始')
            continue
        names = json.loads(mf.read_text())
        done = sum(1 for n in names if (ROOT / ds / f'{n}.done').exists())
        gb = sum((ROOT / ds / n).stat().st_size for n in names if (ROOT / ds / n).exists()) / 1e9
        print(f'{ds}: {done}/{len(names)} 文件完成, 已落盘 {gb:.1f} GB')


def download_file(dest, url):
    """curl 断点续传单文件。返回 'ok' | 'expired' | 'fail'。"""
    r = subprocess.run(
        ['curl', '-sSfL', '-C', '-', '--retry', '5', '--retry-delay', '10',
         '--speed-limit', '10000', '--speed-time', '60',   # <10KB/s 持续 60s 视为死链
         '-o', str(dest), url],
        capture_output=True, text=True)
    if r.returncode == 0:
        return 'ok'
    err = (r.stderr or '')[-200:]
    if r.returncode == 33 or 'HTTP/1.1 416' in err or '416' in err:
        return 'ok'          # 416 = 本地已是完整文件
    if '403' in err or 'AccessDenied' in err or 'expired' in err.lower():
        return 'expired'     # 预签名 URL 过期,需重取
    print(f'    curl rc={r.returncode}: {err}', flush=True)
    return 'fail'


def run():
    rid = get_release()
    print(f'release 锁定: {rid}', flush=True)
    for ds in DATASETS:
        ddir = ROOT / ds
        ddir.mkdir(parents=True, exist_ok=True)
        for round_ in range(200):   # 每轮用一批新鲜 URL;全 done 即退出
            urls = fetch_manifest(rid, ds)
            todo = [(fname_of(u), u) for u in urls if not (ddir / f'{fname_of(u)}.done').exists()]
            if not todo:
                print(f'[{ds}] 全部完成 ({len(urls)} 文件)', flush=True)
                break
            print(f'[{ds}] 第 {round_+1} 轮, 剩 {len(todo)}/{len(urls)} 文件', flush=True)
            refetch = False
            for name, url in todo:
                dest = ddir / name
                t0 = time.time()
                res = download_file(dest, url)
                if res == 'ok':
                    (ddir / f'{name}.done').touch()
                    gb = dest.stat().st_size / 1e9
                    print(f'  ✓ {name} ({gb:.2f} GB, {time.time()-t0:.0f}s)', flush=True)
                elif res == 'expired':
                    print(f'  ↻ {name} URL 过期,重取整批', flush=True)
                    refetch = True
                    break
                else:
                    time.sleep(20)   # 网络故障(如睡眠唤醒),歇口气继续
            if refetch:
                continue
    print('S2-SNAPSHOT-DONE', flush=True)


if __name__ == '__main__':
    if '--status' in sys.argv:
        status()
    else:
        run()
