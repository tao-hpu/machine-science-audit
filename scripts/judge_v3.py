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

# 这些模型在本代理上弃用 temperature(推理模型,发 temperature 会 400);其余一律 temperature=0 保复现
NO_TEMP={'claude-sonnet-5','claude-opus-4-8','claude-opus-4-7'}
# 推理模型的思考 token 计入 max_tokens,预算太小会把正文挤成空串(gemini-2.5-pro 实测
# 2500 全空、8000 正常;sonnet-5 同病,见 2026-07-07 研究日志)
BIG_BUDGET={'gemini-2.5-pro','claude-sonnet-5'}
def mt(model, base):
    return max(base, 8000) if model in BIG_BUDGET else base

def chat(model, prompt, max_tokens=1500):
    body={'model':model,'max_tokens':max_tokens,
          'messages':[{'role':'user','content':prompt}]}
    if model not in NO_TEMP:
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
        except (TimeoutError, urllib.error.URLError, OSError):
            time.sleep(4*(i+1)); continue
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

EVALUATION FACET — SPECIAL RULE (judge the DESIGN, not the object):
Reusing an existing evaluation DESIGN counts as COVERED even when applied to a different object. Concretely, the evaluation facet is COVERED if any prior paper uses the same evaluation design — same benchmark family, same metric semantics, or same validation/statistical framework (e.g. conformal risk control, LLM-as-judge, gradient-signal-then-standard-benchmark) — EVEN IF this contribution swaps in a different benchmark instance, dataset, model, or application object. A mere change of WHAT is measured does not make the evaluation novel; that difference belongs to the DOMAIN facet, not evaluation (do not double-count it here).
Mark evaluation NONE (uncovered) ONLY when the evaluation DESIGN itself is genuinely new: a newly constructed benchmark, a newly defined metric, or a new evaluation/attack protocol that no prior paper used.
Generic experimental moves (ablation, substitution/swap experiments, sensitivity analysis, iso-compute comparison) do NOT by themselves establish evaluation coverage — a prior paper's ablation/substitution counts as matching ONLY if it validated the SAME claim, not merely used the same technique. "Same design" means a NAMED benchmark family, metric, or protocol family — not the generic shape of the experiment ("swap a component, measure accuracy" is a generic shape, not a design).
(Analogy: changing an input parameter is not novelty; changing the internal logic is. Swapping the benchmark/object = new parameter = COVERED. A genuinely new evaluation design = new logic = uncovered.)

DOMAIN FACET — SPECIAL RULE (same anti-mirage logic as evaluation):
The domain facet is COVERED if any prior paper works in the same general application AREA, EVEN IF this contribution targets a different specific sub-problem, dataset, or instance within that area. Applying a known idea to a narrower or specific problem inside an existing area does NOT make the domain novel — that is a change of instance, not of area.
Mark domain NONE (uncovered) ONLY when the contribution operates in a GENUINELY new application area that no prior paper addresses — not merely a new specific instance of an existing area.
(Same parameter-vs-logic test: a new specific problem within a known area = new parameter = COVERED; a genuinely new area = new logic = uncovered.)
Consequence to keep in mind: with evaluation and domain judged this generously, a "novel" verdict should be driven by a genuinely new PURPOSE or MECHANISM, not by the packaging facets. Do not let "applied to my specific setting" alone produce novelty.

PURPOSE FACET — SPECIAL RULE (same parameter-vs-logic test):
The purpose facet is COVERED if any prior paper pursues the SAME OBJECTIVE at the problem-class level — the same gap being closed, the same target quantity being improved — even when the host architecture, model, or application instance differs (that difference belongs to the domain facet).
Positive example: purpose "eliminate wasted memory from fixed cache budgets by sizing the cache to actual need" IS covered by a prior paper on adaptive cache-budget allocation, even if this contribution targets hybrid linear attention and the prior paper targets standard transformers — same objective, different host = COVERED.
Negative example: sharing a TOPIC or problem SPACE is not sharing a purpose. A paper that INTRODUCES an attack does NOT cover the purpose "defend against that attack". A paper about adaptive scheduling in an unrelated field does not cover "show adaptive timing matters for safe-data interleaving" merely because both involve adaptive scheduling.

FINDING-TYPE CONTRIBUTIONS (type "finding") — PER-FACET STANDARDS DIFFER:
A finding's substance is the demonstrated relationship/effect, so apply these standards:
- purpose: covered ONLY if a prior paper reports or establishes the SAME empirical relationship — same variables and direction of effect, any vocabulary. Thematic/conceptual similarity in an unrelated setting does NOT cover.
- mechanism (= the analysis method used to obtain the finding) and evaluation: judge at DESIGN level, exactly like the evaluation rule above — a standard analysis protocol (controlled ablation, multi-seed comparison, per-task breakdown, backtest comparison) that appears in prior work counts as COVERED even if applied to a different benchmark, dataset, or object. Do NOT mark these NONE merely because no prior paper ran the same experiment on this specific object.
- Net effect: for a finding, the high-stakes call is PURPOSE (is the relationship itself new?); mechanism/evaluation should rarely drive novelty.

Now reason step by step. For each facet write 1-2 sentences: which paper number(s), if any, contain an equivalent element, and quote the phrase.

CONTRIBUTION (type: {ctype}):
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

HARD REQUIREMENT for corrections: a correction to COVERED-BY must QUOTE the phrase from that paper containing the equivalent element, and must satisfy the SAME facet-specific standard as the first pass (for a finding's purpose: the same empirical relationship; for evaluation: a named design or the same claim; for purpose: the same objective, not the same topic). "Conceptually similar", "aligns with the concept of", or "could encompass indirectly" are NOT sufficient grounds — if that is all you have, keep NONE.

CONTRIBUTION FACETS IN QUESTION:
{facet_texts}

PRIOR PAPERS:
{papers}

For each questioned facet output one line exactly: FACET: <name> | FINAL: COVERED-BY <n> | or FINAL: NONE | reason: <short>"""

FACETS=('purpose','mechanism','evaluation','domain')
PROMPT_VERSION='v7'  # v7: purpose 锚 + finding 分工 + evaluation 去歧义 + stage2 翻案门槛(2026-07-07)

def parse_verdicts(stage1):
    """Parse the structured 'VERDICT <facet> = <n|NONE>' lines into {facet: int|[int]|None}.
    确定性解析取代原 FINALIZE LLM 调用:旧版把 stage1 截到 3000 字符喂第三轮 LLM,
    冗长模型(sonnet)的 VERDICT 行在末尾被切掉 → 覆盖判定静默变 null → 假 facet-novel。"""
    out={}
    for fac in FACETS:
        m=re.search(rf'VERDICT\s+{fac}\s*=\s*([^\n]*)', stage1, re.I)
        if not m:
            out[fac]='MISSING'; continue
        val=m.group(1)
        nums=[int(x) for x in re.findall(r'\d+', val)]
        if not nums or re.search(r'\bNONE\b', val, re.I):
            out[fac]=None
        else:
            out[fac]=nums[0] if len(nums)==1 else nums
    return out

def parse_none_facets(stage1):
    return [f for f,v in parse_verdicts(stage1).items() if v is None]

def parse_stage2(stage2):
    """Parse 'FACET: <name> | FINAL: COVERED-BY <n>' / 'FINAL: NONE' lines.
    容错:facet 名后跟原文('FACET: domain: LLM safety ... |')、markdown 星号、方括号编号。"""
    out={}
    for m in re.finditer(r'FACET:\s*\**(\w+)[^|\n]*\|\s*\**FINAL\**\s*:?\**\s*(COVERED[- ]BY\s*\[?(\d+)|NONE)', stage2, re.I):
        fac=m.group(1).lower()
        if fac in FACETS:
            out[fac]=int(m.group(3)) if m.group(3) else None
    return out

def judge_one(model, contrib, neighbors, keep_raw=False):
    papers='\n'.join(f"[{i+1}] ({n.get('date')}) {n['title']} — {(n.get('abstract') or '')[:500]}"
                     for i,n in enumerate(neighbors))
    s1=chat(model, STAGE1.format(papers=papers, ctype=contrib.get('type','method'),
                                 **{k:contrib[k] for k in ('purpose','mechanism','evaluation','domain')}), max_tokens=mt(model,2500))
    verdicts=parse_verdicts(s1)
    none=[f for f,v in verdicts.items() if v is None]
    s2=''
    if none:
        facet_texts='\n'.join(f"- {f}: {contrib[f]}" for f in none)
        s2=chat(model, STAGE2.format(none_facets=', '.join(none), facet_texts=facet_texts, papers=papers), max_tokens=mt(model,1500))
        for fac,v in parse_stage2(s2).items():
            if fac in none:  # stage2 只对 NONE facet 有翻案权
                verdicts[fac]=v
    # 三态归类:确定性规则(不再让 LLM 出 single_paper_covers_all)
    fac={f:{'covered_by':(None if verdicts[f]=='MISSING' else verdicts[f])} for f in FACETS}
    if any(v=='MISSING' for v in verdicts.values()):
        state=None  # VERDICT 行缺失 = 解析失败,记 null 不猜
        spa=None
    else:
        sets=[set(v if isinstance(v,list) else [v]) if v is not None else set() for v in verdicts.values()]
        common=set.intersection(*sets) if all(sets) else set()
        spa=min(common) if common else None
        if spa: state='covered'
        else: state='recombination' if all(sets) else 'facet-novel'
    fac['single_paper_covers_all']=spa
    r={'model':model,'state':state,'facets':fac,'self_check_triggered':none}
    if keep_raw:
        r['raw']={'stage1':s1,'stage2':s2}
    return r

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--pid',required=True); ap.add_argument('--cid',default='C1')
    ap.add_argument('--models',default='gpt-4o,claude-sonnet-4-6,gemini-2.5-pro')
    ap.add_argument('--neighbors',default='s2')  # s2|gold|judgment
    a=ap.parse_args()
    contrib=next(c for c in json.load(open(f'data/extractions/{a.pid}.json'))['contributions'] if c['id']==a.cid)
    if a.neighbors=='gold':
        nbrs=json.load(open(f'data/gold_neighbors/{a.pid}.json'))
    elif a.neighbors=='s2':  # retrieve_s2_slow.py 的近邻缓存(正式检索通道)
        j=json.load(open(f'data/neighbors_s2/{a.pid}.json'))
        nbrs=next(c for c in j['contributions'] if c['id']==a.cid)['neighbors']
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
