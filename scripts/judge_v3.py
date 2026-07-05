#!/usr/bin/env python3
"""Facet-coverage judge v3 — step-by-step reasoning + self-verification + multi-judge.

相对 v0.2 的三处升级(回应"prompt 不够顶 + 要 step-by-step 验证"):
1. 逐 facet 显式推理:先让模型对每个 facet 逐篇近邻比对、给出证据,再出结论(不是单轮 JSON)。
2. 自检环节(专治 novelty mirage):对判为 NONE(即驱动 facet-novel 的高风险判定)的 facet,强制第二轮复查"是否真的没有任何一篇、哪怕换个术语也算"。
3. 多裁判:同一输入过 gpt-4o / claude-sonnet-5 / gemini-2.5-pro,报一致性。
1 条 in-context 示例(good-word-attack 跨域匹配)内嵌,教模型把 mechanism 抽象到原语层。
"""
import json, os, re, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load_env():
    e={}
    for l in open(os.path.join(ROOT,'.env')):
        if '=' in l and not l.strip().startswith('#'):
            k,v=l.split('=',1); e[k.strip()]=v.strip().strip('"').strip("'")
    return e
ENV=load_env()

def chat(model, prompt, max_tokens=1500):
    body={'model':model,'max_tokens':max_tokens,
          'messages':[{'role':'user','content':prompt}]}
    if not model.startswith('claude'):  # claude-sonnet-5 rejects temperature on this proxy
        body['temperature']=0
    for i in range(4):
        try:
            req=urllib.request.Request(f"{ENV['LLM_API_BASE']}/chat/completions", data=json.dumps(body).encode(),
                headers={'Authorization':f"Bearer {ENV['LLM_API_KEY']}",'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=150) as r:
                return json.load(r)['choices'][0]['message']['content']
        except urllib.error.HTTPError as e:
            if e.code in (429,503): time.sleep(6*(i+1)); continue
            raise
    raise RuntimeError('chat failed')

def parse_json(text):
    """Strip code fences (gemini) and pull the JSON object."""
    text=re.sub(r'```(?:json)?','',text)
    m=re.search(r'\{.*\}', text, re.S)
    return json.loads(m.group(0)) if m else None

def covered(v):
    """covered_by may be int, list, or null. Any non-empty value = covered."""
    if v is None: return False
    if isinstance(v,list): return len(v)>0
    return True

STAGE1 = """You are auditing whether a research contribution recombines existing ideas or introduces something genuinely absent from prior work. This is a FACTUAL coverage check, NOT a quality or novelty rating.

A contribution is decomposed into four facets:
- purpose (the objective/problem)
- mechanism (the technical primitive/approach)
- evaluation (the validation setup)
- domain (the application area)

For EACH facet, examine EVERY prior paper below and decide whether any contains a SUBSTANTIVELY EQUIVALENT element. Equivalence is at the level of the underlying primitive: different names, notation, scale, or application domain do NOT break equivalence. In particular, judge the MECHANISM cross-domain — the same primitive under a different field's vocabulary still counts.

WORKED EXAMPLE (how to reason):
Contribution mechanism = "prepend/append benign text to harmful content to dilute the signal so a safety classifier misses it."
Prior paper = "Good Word Attacks on Statistical Spam Filters" (adds innocuous words to spam so a statistical filter scores it as ham).
Correct call: mechanism COVERED — both are "dilute a classifier's decision signal by padding with benign tokens," the same primitive despite spam≠LLM-safety. Do NOT mark it novel just because the application (LLM safety judge) is newer.

Now reason step by step. For each facet write 1-2 sentences: which paper number(s), if any, contain an equivalent element, and quote the phrase.

CONTRIBUTION:
- purpose: {purpose}
- mechanism: {mechanism}
- evaluation: {evaluation}
- domain: {domain}

PRIOR PAPERS:
{papers}

Write your per-facet reasoning first (purpose, mechanism, evaluation, domain in order). Then end with EXACTLY four verdict lines in this format and nothing after them:
VERDICT purpose = <paper number or NONE>
VERDICT mechanism = <paper number or NONE>
VERDICT evaluation = <paper number or NONE>
VERDICT domain = <paper number or NONE>"""

STAGE2 = """You previously judged some facets as NOT covered by any prior paper. Marking a facet as uncovered is the high-stakes call — it is the claim that drives a "novel" verdict, and it is easy to over-claim by missing a prior work that uses different terminology.

Re-examine ONLY these facets you marked NONE: {none_facets}

For each, scan the prior papers ONE more time. Ask: is there truly no paper containing this element even under a DIFFERENT name, an earlier framing, or another domain's vocabulary? If you now find one, correct it. If still none, confirm.

CONTRIBUTION FACETS IN QUESTION:
{facet_texts}

PRIOR PAPERS:
{papers}

For each questioned facet output one line exactly: FACET: <name> | FINAL: COVERED-BY <n> | or FINAL: NONE | reason: <short>"""

FINALIZE = """Convert the analysis into strict JSON. Use the FINAL decisions from the re-examination where present, otherwise the first-pass COVERED-BY.

First pass:
{stage1}

Re-examination:
{stage2}

Output JSON only:
{{"purpose":{{"covered_by":<int|null>}},"mechanism":{{"covered_by":<int|null>}},"evaluation":{{"covered_by":<int|null>}},"domain":{{"covered_by":<int|null>}},"single_paper_covers_all":<int|null>}}"""

def parse_none_facets(stage1):
    """Parse the structured 'VERDICT <facet> = <n|NONE>' lines. Only these decide the self-check trigger."""
    none=[]
    for fac in ('purpose','mechanism','evaluation','domain'):
        m=re.search(rf'VERDICT\s+{fac}\s*=\s*(NONE|\d+)', stage1, re.I)
        if m and m.group(1).upper()=='NONE': none.append(fac)
    return none

def judge_one(model, contrib, neighbors):
    papers='\n'.join(f"[{i+1}] ({n.get('date')}) {n['title']} — {(n.get('abstract') or '')[:500]}"
                     for i,n in enumerate(neighbors))
    s1=chat(model, STAGE1.format(papers=papers, **{k:contrib[k] for k in ('purpose','mechanism','evaluation','domain')}))
    none=parse_none_facets(s1)
    s2=''
    if none:
        facet_texts='\n'.join(f"- {f}: {contrib[f]}" for f in none)
        s2=chat(model, STAGE2.format(none_facets=', '.join(none), facet_texts=facet_texts, papers=papers))
    fin=chat(model, FINALIZE.format(stage1=s1[:3000], stage2=s2[:1500]), 400)
    fac=parse_json(fin)
    state=None
    if fac:
        if covered(fac.get('single_paper_covers_all')): state='covered'
        else:
            cov=[covered(fac[x].get('covered_by')) for x in ('purpose','mechanism','evaluation','domain')]
            state='recombination' if all(cov) else 'facet-novel'
    return {'model':model,'state':state,'facets':fac,'self_check_triggered':none}

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--pid',required=True); ap.add_argument('--cid',default='C1')
    ap.add_argument('--models',default='gpt-4o,claude-sonnet-5,gemini-2.5-pro')
    ap.add_argument('--neighbors',default='gold')  # gold|judgment
    a=ap.parse_args()
    contrib=next(c for c in json.load(open(f'data/extractions/{a.pid}.json'))['contributions'] if c['id']==a.cid)
    if a.neighbors=='gold':
        nbrs=json.load(open(f'data/gold_neighbors/{a.pid}.json'))
    else:
        j=json.load(open(f'data/judgments/{a.pid}.json'))
        nbrs=next(c for c in j['contributions'] if c['id']==a.cid)['neighbors']
    print(f'{a.pid}/{a.cid} | {len(nbrs)} neighbors | mechanism: {contrib["mechanism"][:70]}')
    for model in a.models.split(','):
        try:
            r=judge_one(model, contrib, nbrs)
            f=r['facets']
            if not f:
                print(f'  {model:20s} -> PARSE-FAIL (no JSON)'); continue
            g=lambda k: f.get(k,{}).get('covered_by')
            print(f'  {model:20s} -> {str(r["state"]):14s} facets={{p:{g("purpose")},m:{g("mechanism")},e:{g("evaluation")},d:{g("domain")}}} selfcheck={r["self_check_triggered"]}')
        except Exception as e:
            print(f'  {model:20s} -> ERROR {repr(e)[:80]}')
