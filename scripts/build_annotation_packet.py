#!/usr/bin/env python3
"""人工校准标注包生成器:抽样贡献 → 自包含盲评 HTML(标注者那份) + 还原 key(仅 Tao 留)。

盲评保证:HTML 里不含来源系统、不含 pipeline 自动判定、不含 pid 真名(用 opaque item_id)。
key.json 把 item_id 映射回 (arm,pid,cid,auto_state) 供收回后 join + 算一致性/kappa。

抽样:按三态分层,facet-novel 过采样(争议类),默认 60 条机器臂。
用法:python3 scripts/build_annotation_packet.py --arm machine --n 60 --out data/annotation
"""
import argparse, json, glob, os, re, sys, html, hashlib

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())   # 让 `import scripts.judge_v3` 可用(翻译人类中文 facet 时)

ARMS = {
    'machine': ('data/judgments_fars_rescue', 'data/extractions',        'data/neighbors_fars_cito_rescue'),
    'human':   ('data/judgments_human_rescue', 'data/extractions_human', 'data/neighbors_human_rescue'),
}
FACETS = ('purpose', 'mechanism', 'evaluation', 'domain')

# 人类抽取部分为中文;混合盲评要求全英文(否则中英混排一眼看穿人机身份,破盲)。
# 逐 facet 检测 CJK → LLM 翻译成英文,缓存到 data/annotation/_trans_cache.json(确定性、可复现)。
_CJK = re.compile(r'[一-鿿]')
_TRANS_CACHE_PATH = 'data/annotation/_trans_cache.json'
_trans_cache = None

def _translate(text):
    global _trans_cache
    if not text or not _CJK.search(text):
        return text
    if _trans_cache is None:
        _trans_cache = json.load(open(_TRANS_CACHE_PATH)) if os.path.exists(_TRANS_CACHE_PATH) else {}
    key = hashlib.md5(text.encode()).hexdigest()
    if key in _trans_cache:
        return _trans_cache[key]
    import scripts.judge_v3 as J
    prompt = ('Translate this research-contribution facet to concise academic English. '
              'Output ONLY the translation, no quotes, no preamble.\n\n' + text)
    out = J.chat('gpt-4o', prompt, max_tokens=300).strip()
    _trans_cache[key] = out
    os.makedirs('data/annotation', exist_ok=True)
    json.dump(_trans_cache, open(_TRANS_CACHE_PATH, 'w'), ensure_ascii=False, indent=1)
    return out


def load_contrib(ext, pid, cid):
    for c in json.load(open(f'{ext}/{pid}.json'))['contributions']:
        if c['id'] == cid:
            return c
    return None


def load_neighbors(nbr, pid, cid):
    for c in json.load(open(f'{nbr}/{pid}.json'))['contributions']:
        if c['id'] == cid:
            return c.get('neighbors', [])
    return []


def sample(jdir, n):
    by_state = {'facet-novel': [], 'recombination': [], 'covered': []}
    for f in glob.glob(f'{jdir}/*.json'):
        d = json.load(open(f)); pid = d['paper_id']
        for c in d['contributions']:
            st = c.get('consensus_state_v2') or c.get('consensus_state')
            if st in by_state:
                by_state[st].append((pid, c['cid'], st))
    # 分层配额:facet-novel 过采样(争议类)
    quota = {'facet-novel': n // 2, 'recombination': n // 4, 'covered': n - n // 2 - n // 4}
    picked = []
    for st, q in quota.items():
        lst = sorted(by_state[st]); step = max(1, len(lst) // max(1, q))
        picked += lst[::step][:q]
    return picked


def build(arm, n, outdir):
    os.makedirs(outdir, exist_ok=True)
    # mixed:两臂各半分层抽样,合并打散 → 真盲(标注者无从分辨人机)
    if arm == 'mixed':
        picked = [('machine',) + t for t in sample(ARMS['machine'][0], n // 2)] \
               + [('human',) + t for t in sample(ARMS['human'][0], n - n // 2)]
    else:
        picked = [(arm,) + t for t in sample(ARMS[arm][0], n)]
    items, key = [], {}
    for src, pid, cid, auto in picked:
        _, ext, nbr = ARMS[src]
        c = load_contrib(ext, pid, cid); nb = load_neighbors(nbr, pid, cid)
        if not c or not nb:
            continue
        iid = 'I' + hashlib.md5(f'{src}/{pid}/{cid}'.encode()).hexdigest()[:6].upper()
        key[iid] = {'arm': src, 'pid': pid, 'cid': cid, 'auto_state': auto}
        items.append({
            'iid': iid,
            'type': c.get('type', ''),
            'facets': {k: _translate((c.get(k) or '').strip()) for k in FACETS},
            'neighbors': [{'n': j + 1, 'title': h['title'],
                           'abstract': (h.get('abstract') or '')[:450]}
                          for j, h in enumerate(nb[:15])],
        })
    # 挂 LLM 元审提示(若已生成):默认折叠、答后才揭示,只服务讨论/仲裁,不锚定独立标注
    hpath = os.path.join(outdir, f'hints_{arm}.json')
    hints = json.load(open(hpath)) if os.path.exists(hpath) else {}
    for it in items:
        it['hint'] = hints.get(it['iid'])
    # 打散顺序(盲:不按三态聚堆),确定性种子=iid 排序
    items.sort(key=lambda x: x['iid'])
    html_path = os.path.join(outdir, f'annotation_packet_{arm}.html')
    key_path = os.path.join(outdir, f'annotation_key_{arm}.json')
    open(key_path, 'w').write(json.dumps(key, ensure_ascii=False, indent=1))
    open(html_path, 'w').write(render(items, arm))
    print(f'✅ {len(items)} 条 → {html_path}')
    print(f'🔑 还原 key(仅你留)→ {key_path}')
    dist = {}
    for v in key.values():
        dist[v['auto_state']] = dist.get(v['auto_state'], 0) + 1
    print(f'   抽样三态分布(自动判定,标注者看不到):{dist}')


def render(items, arm):
    data = json.dumps(items, ensure_ascii=False)
    esc = html.escape
    return r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Novelty Annotation Packet</title>
<style>
:root{--bg:#faf9f7;--fg:#1a1a1a;--mut:#666;--bd:#ddd;--card:#fff;--accent:#3a6ea5;--warn:#b5651d}
*{box-sizing:border-box}body{margin:0;font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,"PingFang SC","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--fg)}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--bd);padding:12px 20px;z-index:10;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
header h1{font-size:16px;margin:0}
#prog{font-size:13px;color:var(--mut)}
button{font:inherit;padding:7px 14px;border:1px solid var(--accent);background:var(--accent);color:#fff;border-radius:6px;cursor:pointer}
button.sec{background:#fff;color:var(--accent)}
.wrap{max-width:920px;margin:0 auto;padding:20px}
details.guide{background:#fff;border:1px solid var(--bd);border-radius:8px;padding:10px 14px;margin-bottom:18px}
details.guide summary{cursor:pointer;font-weight:600}
.guide h4{margin:12px 0 4px}.guide p,.guide li{font-size:13.5px;color:#333}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:18px;margin-bottom:22px}
.iid{font:12px monospace;color:var(--mut)}
.facet{margin:6px 0;padding:8px 10px;background:#f4f2ee;border-radius:6px}
.facet b{display:inline-block;min-width:92px;color:var(--accent)}
.nbrs{margin:12px 0;border-top:1px dashed var(--bd);padding-top:10px}
.nbr{font-size:13px;margin:5px 0;padding-left:6px}
.nbr .num{display:inline-block;min-width:26px;font-weight:700;color:var(--warn)}
.nbr .ab{color:var(--mut)}
.form{margin-top:14px;border-top:2px solid var(--bd);padding-top:12px}
.row{display:flex;gap:10px;align-items:center;margin:7px 0;flex-wrap:wrap}
.row label{min-width:130px;font-weight:600;font-size:13.5px}
input[type=text]{font:inherit;padding:5px 8px;border:1px solid var(--bd);border-radius:5px;width:120px}
input.wide{width:100%;max-width:520px}
.state label{min-width:auto;font-weight:400;margin-right:14px}
textarea{font:inherit;width:100%;max-width:640px;min-height:48px;padding:6px 8px;border:1px solid var(--bd);border-radius:5px}
.hint{font-size:12px;color:var(--mut)}
.done{border-color:#5a9;box-shadow:0 0 0 2px #5a92}
.hintbox{margin-top:10px;border-top:1px dashed var(--bd);padding-top:8px}
.hintbtn{background:#fff;color:var(--warn);border-color:var(--warn);font-size:13px;padding:5px 10px}
.hinttext{margin-top:8px;padding:8px 10px;background:#fff7ee;border:1px solid var(--warn);border-radius:6px;font-size:13px}
.hint-note{margin-top:6px;font-size:11.5px;color:var(--mut);font-style:italic}
@media(prefers-color-scheme:dark){.hinttext{background:#332a1e}.hintbtn{background:#26251f}}
@media(prefers-color-scheme:dark){:root{--bg:#1c1b19;--fg:#eee;--mut:#aaa;--bd:#444;--card:#26251f;--accent:#7ab0e0}.facet{background:#333127}input,textarea{background:#1c1b19;color:#eee}}
</style></head><body>
<header>
 <h1>Novelty Annotation</h1>
 <span id="prog">0 / 0</span>
 <button class="sec" onclick="save()">Save progress</button>
 <button onclick="exp()">Export answers (JSON)</button>
 <span class="hint">Answers auto-save in this browser. Export when done and send the file back.</span>
</header>
<div class="wrap">
<details class="guide"><summary>▶ Instructions & rubric (read first)</summary>
<h4>Your task</h4><p>For each contribution you see its four <b>facets</b> and 15 retrieved <b>prior papers</b> (all published before this contribution). Decide which of three states it is in, <b>using only the papers shown</b> — do not use your own memory of the literature (if you know an earlier work not listed, note it in Notes; that's a retrieval issue, not your call).</p>
<h4>Facet coverage = substantive equivalence</h4><p>A facet is "covered" by a prior paper if the core idea is the same — different name/symbol/scale still counts as equivalent; only a different core idea counts as different.</p>
<ul>
<li><b>purpose covered</b> = a prior paper pursues the same objective (same gap / same target quantity), even if the host model/application differs. Sharing only a topic or problem-space is NOT the same purpose. (An attack paper does NOT cover the purpose "defend against that attack".)</li>
<li><b>mechanism covered</b> = same technical primitive, abstracted past the application skin.</li>
<li><b>evaluation covered</b> = judge by <i>design</i>, not claim: only a genuinely new benchmark / new metric / new protocol is uncovered; swapping the dataset/model being tested = covered.</li>
<li><b>domain covered</b> = a prior paper is in the same general application area, even if this targets a narrower sub-problem; only a wholly new area is uncovered.</li>
</ul>
<h4>Three states</h4>
<ul>
<li><b>Covered</b>: a <i>single</i> prior paper is substantively equivalent on all four facets.</li>
<li><b>Recombination</b>: every facet is found somewhere in the set, but <i>no single paper</i> covers all four.</li>
<li><b>Facet-novel</b>: at least one facet has no equivalent in any of the 15 papers (mark which).</li>
</ul>
<p class="hint">You are blind to the paper's source and to any automated verdict. For each facet, record the covering paper number (or NONE), then pick the state. Every verdict needs an evidence note citing paper numbers.</p>
</details>
<div id="cards"></div>
<div style="text-align:center;margin:30px 0"><button onclick="exp()">Export answers (JSON)</button></div>
</div>
<script>
const ITEMS = ''' + data + r''';
const KEY='novelty_annot_'+(ITEMS[0]?ITEMS[0].iid:'x');
let ans = JSON.parse(localStorage.getItem(KEY)||'{}');
const cards=document.getElementById('cards');
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
ITEMS.forEach(it=>{
 const a=ans[it.iid]||{};
 const c=document.createElement('div');c.className='card';c.id='c_'+it.iid;
 let fh='';for(const[k,v]of Object.entries(it.facets)){if(v)fh+=`<div class="facet"><b>${k}</b> ${esc(v)}</div>`}
 let nh='';it.neighbors.forEach(n=>{nh+=`<div class="nbr"><span class="num">[${n.n}]</span>${esc(n.title)}. <span class="ab">${esc(n.abstract)}</span></div>`});
 c.innerHTML=`<div class="iid">${it.iid} · type: ${esc(it.type)}</div>${fh}
 <div class="nbrs"><b>Prior papers (retrieved):</b>${nh}</div>
 <div class="form">
  <div class="row"><label>Coverage (paper # or NONE):</label>
   P<input type="text" data-i="${it.iid}" data-f="P" value="${a.P||''}" style="width:70px">
   M<input type="text" data-i="${it.iid}" data-f="M" value="${a.M||''}" style="width:70px">
   E<input type="text" data-i="${it.iid}" data-f="E" value="${a.E||''}" style="width:70px">
   D<input type="text" data-i="${it.iid}" data-f="D" value="${a.D||''}" style="width:70px"></div>
  <div class="row state"><label>State:</label>
   ${['covered','recombination','facet-novel'].map(s=>`<label><input type="radio" name="st_${it.iid}" data-i="${it.iid}" data-f="state" value="${s}" ${a.state===s?'checked':''}> ${s}</label>`).join('')}</div>
  <div class="row"><label>Evidence (cite #):</label><input class="wide" type="text" data-i="${it.iid}" data-f="evidence" value="${esc(a.evidence||'')}"></div>
  <div class="row"><label>Confidence:</label>
   ${['high','med','low'].map(s=>`<label><input type="radio" name="cf_${it.iid}" data-i="${it.iid}" data-f="confidence" value="${s}" ${a.confidence===s?'checked':''}> ${s}</label>`).join('')}</div>
  <div class="row"><label>Notes:</label><textarea data-i="${it.iid}" data-f="notes">${esc(a.notes||'')}</textarea></div>
 </div>
 ${it.hint?`<div class="hintbox">
   <button type="button" class="hintbtn" onclick="reveal('${it.iid}')">🔒 Reveal AI hint — only AFTER you've chosen a State above</button>
   <div class="hinttext" id="h_${it.iid}" style="display:none">
     <b>AI meta-check:</b> ${it.hint.level==='suspect'?'⚠ SUSPECT — a human should double-check this':'✓ likely solid'} · confidence: ${esc(it.hint.confidence||'')}<br>
     <b>Check:</b> ${esc(it.hint.checkpoint||'')}
     <div class="hint-note">This is only to focus group discussion. Your recorded answer above stays as-is — do not change it to match the AI.</div>
   </div></div>`:''}`;
 cards.appendChild(c);
});
function reveal(iid){
 const picked=document.querySelector('input[name="st_'+iid+'"]:checked');
 if(!picked){alert('Record your own State for this item first — then reveal the AI hint (for discussion only).');return;}
 document.getElementById('h_'+iid).style.display='block';
}
function upd(e){const t=e.target;if(!t.dataset.i)return;const i=t.dataset.i,f=t.dataset.f;
 ans[i]=ans[i]||{};ans[i][f]=t.value;save();mark();}
document.addEventListener('input',upd);document.addEventListener('change',upd);
function mark(){let done=0;ITEMS.forEach(it=>{const a=ans[it.iid]||{};const ok=a.state&&a.evidence;
 const el=document.getElementById('c_'+it.iid);if(el)el.classList.toggle('done',!!ok);if(ok)done++});
 document.getElementById('prog').textContent=done+' / '+ITEMS.length+' done';}
function save(){localStorage.setItem(KEY,JSON.stringify(ans));}
function exp(){const blob=new Blob([JSON.stringify({packet:KEY,answers:ans},null,1)],{type:'application/json'});
 const u=URL.createObjectURL(blob);const a=document.createElement('a');a.href=u;a.download='answers_'+KEY+'.json';a.click();}
mark();
</script></body></html>'''


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['machine', 'human', 'mixed'], default='mixed')
    ap.add_argument('--n', type=int, default=72)
    ap.add_argument('--out', default='data/annotation')
    a = ap.parse_args()
    build(a.arm, a.n, a.out)
