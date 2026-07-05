#!/usr/bin/env python3
"""Download ICAIS 2025 AI-generated papers from AiraXiv (research use; polite crawl)."""
import urllib.request, re, os, time, json

BASE = '/Users/TaoTao/Desktop/Learn/硕士阶段/machine-science-audit/data/airaxiv-icais2025'
os.makedirs(BASE, exist_ok=True)
UA = {'User-Agent': 'novelty-audit-research/0.1 (mailto:tan1@my.hpu.edu; academic corpus study)'}

ids = set()
for page in range(1, 10):
    url = f'https://airaxiv.com/papers/?conference=ICAIS_2025&type=ai_generated&page={page}'
    try:
        req = urllib.request.Request(url, headers=UA)
        html = urllib.request.urlopen(req, timeout=30).read().decode('utf-8', 'ignore')
    except Exception as e:
        print(f'page {page}: {e}'); break
    found = set(re.findall(r'/papers/(?:view|pdf)/([0-9]{4}\.[0-9]{4,5})', html))
    if not found:
        print(f'page {page}: no ids, stopping'); break
    before = len(ids); ids |= found
    print(f'page {page}: +{len(ids)-before} ids (total {len(ids)})', flush=True)
    time.sleep(1.2)

manifest = []
for pid in sorted(ids):
    dest = os.path.join(BASE, f'{pid}.pdf')
    if os.path.exists(dest) and os.path.getsize(dest) > 10000:
        manifest.append(pid); continue
    try:
        req = urllib.request.Request(f'https://airaxiv.com/papers/pdf/{pid}/', headers=UA)
        data = urllib.request.urlopen(req, timeout=60).read()
        if data[:4] == b'%PDF':
            open(dest, 'wb').write(data); manifest.append(pid)
        else:
            print(f'{pid}: not a PDF ({len(data)} bytes)')
    except Exception as e:
        print(f'{pid}: {e}')
    time.sleep(1.2)

json.dump(sorted(manifest), open(os.path.join(BASE, 'manifest.json'), 'w'), indent=1)
print(f'DONE: {len(manifest)} PDFs in {BASE}')
