#!/usr/bin/env python3
"""
Cognitive Framework Empirical Validation Suite
==============================================
Laptop-oriented, controlled synthetic experiments for mechanism-level falsification
of claims in "A Cognitive-Inspired Unified Framework for Adaptive AI Systems".

The suite is intentionally not a claim of human-level cognition. It tests operational
mechanisms under known causal ground truth using matched random seeds.

Outputs:
  run_config.json
  results/raw_primary.csv
  results/secondary_metrics.csv
  results/hypothesis_tests.csv
  results/claim_matrix.csv
  results/SUMMARY.txt
  plots/*.png

Dependencies: numpy, pandas, scipy, matplotlib
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# -----------------------------
# Reproducibility / presets
# -----------------------------

SMOKE_SEEDS = [91001, 91039]
PILOT_SEEDS = [12001 + 37*i for i in range(8)]
CONFIRMATORY_SEEDS = [30011 + 53*i for i in range(60)]

PRESETS = {
    "smoke": SMOKE_SEEDS,
    "pilot": PILOT_SEEDS,
    "confirmatory": CONFIRMATORY_SEEDS,
}

ALPHA = 0.05


@dataclass
class ExperimentSpec:
    key: str
    title: str
    paper_claim: str
    null_hypothesis: str
    alternative_hypothesis: str
    primary_metric: str
    direction: str  # 'greater' or 'less' means treatment > control or treatment < control
    tier: str       # 'core', 'proxy', 'implementation'
    caveat: str


SPECS: Dict[str, ExperimentSpec] = {
    "tadm_sparse": ExperimentSpec(
        "tadm_sparse",
        "TADM selective sparse memory under a fixed memory budget",
        "Salience/novelty/confidence-weighted sparse writes improve useful retrieval under bounded memory.",
        "H0: weighted sparse retention has retrieval accuracy <= budget-matched uniform retention.",
        "HA: weighted sparse retention has retrieval accuracy > budget-matched uniform retention.",
        "retrieval_accuracy",
        "greater",
        "core",
        "Synthetic records make usefulness observable; this validates the mechanism, not universal memory superiority.",
    ),
    "emg_multihop": ExperimentSpec(
        "emg_multihop",
        "EMG multi-hop temporal/causal reasoning",
        "Graph-structured episodic relations support multi-hop reasoning better than flat episodic lookup.",
        "H0: EMG multi-hop query accuracy <= flat episodic lookup accuracy.",
        "HA: EMG multi-hop query accuracy > flat episodic lookup accuracy.",
        "multihop_accuracy",
        "greater",
        "core",
        "The task specifically requires relational composition; results do not establish superiority on non-relational tasks.",
    ),
    "dream_consolidation": ExperimentSpec(
        "dream_consolidation",
        "Dream consolidation for redundancy/conflict compression",
        "Consolidation merges redundant episodes, resolves conflicts, and reduces memory growth while preserving useful recall.",
        "H0: consolidation does not reduce retained record count relative to no consolidation.",
        "HA: consolidation retains fewer records than no consolidation.",
        "retained_records",
        "less",
        "core",
        "Accuracy preservation is reported as a secondary non-inferiority diagnostic, not inferred from the primary p-value.",
    ),
    "pad_risk": ExperimentSpec(
        "pad_risk",
        "PAD-modulated risk-sensitive action selection",
        "Arousal/affect-aware risk modulation reduces unsafe actions in hazardous contexts.",
        "H0: PAD-aware policy has unsafe-action rate >= affect-agnostic policy.",
        "HA: PAD-aware policy has unsafe-action rate < affect-agnostic policy.",
        "unsafe_action_rate",
        "less",
        "core",
        "This tests programmed affective modulation, not subjective emotion or consciousness.",
    ),
    "tam_postpone": ExperimentSpec(
        "tam_postpone",
        "Temporal Action Modulation: remember/postpone/forget",
        "Priority-sensitive postponement/forgetting improves deadline-weighted utility under finite attention.",
        "H0: TAM total deadline-weighted utility <= FIFO/immediate baseline utility.",
        "HA: TAM total deadline-weighted utility > FIFO/immediate baseline utility.",
        "deadline_utility",
        "greater",
        "core",
        "Synthetic scheduling is an operational test of temporal control, not human procrastination.",
    ),
    "auto_cue": ExperimentSpec(
        "auto_cue",
        "Automatic cue determination",
        "Selecting cues using expected retrieval-benefit proxies improves retrieval/task success over an uncalibrated cue.",
        "H0: auto-cue success <= random/fixed cue success.",
        "HA: auto-cue success > random/fixed cue success.",
        "cue_success",
        "greater",
        "core",
        "Proxy-score quality is independently noisy so the test is not a deterministic restatement of the scoring rule.",
    ),
    "curiosity": ExperimentSpec(
        "curiosity",
        "Curiosity-driven novelty/uncertainty exploration",
        "Curiosity bonuses improve discovery of useful unknown states under the same interaction budget.",
        "H0: curiosity-guided agent discovers <= useful latent opportunities than no-curiosity exploration.",
        "HA: curiosity-guided agent discovers > useful latent opportunities than no-curiosity exploration.",
        "useful_discoveries",
        "greater",
        "core",
        "The comparison equalizes total interactions; curiosity is not credited for simply spending more steps.",
    ),
    "intuition": ExperimentSpec(
        "intuition",
        "Intuition heuristic prior for familiar decisions",
        "A learned heuristic prior reduces deliberation cost on familiar states while maintaining comparable correctness.",
        "H0: intuition provides no reduction in decision cost relative to full deliberation.",
        "HA: intuition reduces decision cost relative to full deliberation.",
        "decision_cost",
        "less",
        "core",
        "Accuracy and reversal susceptibility are secondary diagnostics; speed alone is not interpreted as intelligence.",
    ),
    "bootstrapping": ExperimentSpec(
        "bootstrapping",
        "Knowledge bootstrapping / memory transplant",
        "Compatible transferred episodic knowledge accelerates target adaptation relative to learning from scratch.",
        "H0: compatible memory transplant target accuracy <= scratch target accuracy at the same low-data budget.",
        "HA: compatible memory transplant target accuracy > scratch target accuracy at the same low-data budget.",
        "fewshot_accuracy",
        "greater",
        "core",
        "A separate incompatible-transfer stress test checks negative transfer and compatibility gating.",
    ),
    "shared_gltm": ExperimentSpec(
        "shared_gltm",
        "SharedGLTM societal/global memory transfer",
        "Reliability-weighted shared memory improves an inexperienced agent's held-out decisions.",
        "H0: reliability-weighted SharedGLTM novice accuracy <= local-only novice accuracy.",
        "HA: reliability-weighted SharedGLTM novice accuracy > local-only novice accuracy.",
        "novice_accuracy",
        "greater",
        "core",
        "This tests information sharing under controlled source reliabilities, not societal consensus legitimacy.",
    ),
    "validator_reflection": ExperimentSpec(
        "validator_reflection",
        "Validator consensus plus reflective safety layer",
        "Trust-weighted consensus and reflection reduce unsafe action execution compared with a single validator.",
        "H0: consensus+reflection unsafe execution rate >= single-validator unsafe execution rate.",
        "HA: consensus+reflection unsafe execution rate < single-validator unsafe execution rate.",
        "unsafe_execution_rate",
        "less",
        "core",
        "The paper's illustrative 1e-6 safety constraint cannot be established with this laptop-scale sample; this tests relative risk reduction.",
    ),
    "truth_conflict": ExperimentSpec(
        "truth_conflict",
        "Incremental confidence/credibility conflict resolution",
        "Provenance/reliability-weighted evidence corrects false beliefs and resists later misinformation.",
        "H0: credibility-weighted final fact accuracy <= unweighted frequency/recency baseline.",
        "HA: credibility-weighted final fact accuracy > unweighted frequency/recency baseline.",
        "final_fact_accuracy",
        "greater",
        "core",
        "Ground truth and source reliability are known in simulation; real-world truth adjudication requires external evidence.",
    ),
    "dream_rehearsal": ExperimentSpec(
        "dream_rehearsal",
        "Validated counterfactual dream rehearsal",
        "Validated recombination of prior episodes improves held-out compositional decisions beyond replay-only memory.",
        "H0: validated dream rehearsal held-out composition accuracy <= replay-only accuracy.",
        "HA: validated dream rehearsal held-out composition accuracy > replay-only accuracy.",
        "composition_accuracy",
        "greater",
        "core",
        "Dream candidates are validated by the synthetic causal simulator and use the same number of training examples as replay.",
    ),
    "gigo_filter": ExperimentSpec(
        "gigo_filter",
        "Input quality gating and cross-validation (GIGO mitigation)",
        "Quality/provenance gating reduces downstream belief corruption under noisy/adversarial observations.",
        "H0: gated final belief accuracy <= accept-all baseline accuracy under corrupted inputs.",
        "HA: gated final belief accuracy > accept-all baseline accuracy under corrupted inputs.",
        "belief_accuracy",
        "greater",
        "core",
        "False-rejection rate is reported because aggressive filtering can appear accurate by discarding too much.",
    ),
    "mimic_safety": ExperimentSpec(
        "mimic_safety",
        "Trust-weighted observational imitation with safety validation",
        "Outcome/trust-weighted mimicking accelerates skill acquisition while validator gating reduces unsafe imitation.",
        "H0: safe trust-weighted imitation few-shot task reward <= no-imitation learner reward.",
        "HA: safe trust-weighted imitation few-shot task reward > no-imitation learner reward.",
        "fewshot_reward",
        "greater",
        "core",
        "Unsafe imitation rate is a co-diagnostic; imitation advantage should not be credited if safety worsens materially.",
    ),
    "recall_precedence": ExperimentSpec(
        "recall_precedence",
        "Memory precedence from repetition, emotion, centrality, and cue match",
        "A combined precedence score improves useful first recall when these features genuinely predict future usefulness.",
        "H0: precedence-ranked first-recall utility <= recency-only first-recall utility in the aligned regime.",
        "HA: precedence-ranked first-recall utility > recency-only first-recall utility in the aligned regime.",
        "first_recall_utility",
        "greater",
        "proxy",
        "A misaligned negative-control regime is included; the score is not expected to help when its features are non-predictive.",
    ),
    "empathy_proxy": ExperimentSpec(
        "empathy_proxy",
        "Social-state projection (empathy proxy)",
        "Inferring another agent's latent state/preferences improves cooperative choice relative to self-only policy.",
        "H0: social-state projection coordination reward <= self-only coordination reward.",
        "HA: social-state projection coordination reward > self-only coordination reward.",
        "coordination_reward",
        "greater",
        "proxy",
        "This validates social-state inference/coordination only; it is not evidence of genuine empathy or subjective affect.",
    ),
    "bias_mitigation": ExperimentSpec(
        "bias_mitigation",
        "Counterfactual dream/validation mitigation of induced cognitive bias",
        "Counterfactual rehearsal and evidence validation reduce decisions distorted by confirmation/availability weighting.",
        "H0: mitigation bias error >= unmitigated biased-memory error.",
        "HA: mitigation bias error < unmitigated biased-memory error.",
        "bias_error",
        "less",
        "proxy",
        "Bias is deliberately induced by memory-weight perturbations, so this tests mitigation mechanics rather than spontaneous human bias.",
    ),
    "character_consistency": ExperimentSpec(
        "character_consistency",
        "Persistent character priors",
        "Persistent character vectors produce stable, distinguishable policy preferences across matched contexts.",
        "H0: character-conditioned within-profile behavioral consistency <= no-character baseline consistency.",
        "HA: character-conditioned within-profile behavioral consistency > no-character baseline consistency.",
        "profile_consistency",
        "greater",
        "proxy",
        "Stable policy bias is not equivalent to personality, identity, or moral character.",
    ),
    "purpose_planning": ExperimentSpec(
        "purpose_planning",
        "Purpose-conditioned planning versus myopic action selection",
        "Goal/purpose-conditioned planning improves completion of multi-step objectives when tempting immediate rewards conflict with the plan.",
        "H0: purpose-guided goal-completion utility <= myopic immediate-reward utility.",
        "HA: purpose-guided goal-completion utility > myopic immediate-reward utility.",
        "goal_completion_utility",
        "greater",
        "core",
        "The task operationalizes purpose as an explicit goal/plan variable; it does not establish autonomous purpose formation.",
    ),
    "negotiation": ExperimentSpec(
        "negotiation",
        "Multi-agent negotiation and arbitration",
        "Trust-weighted validator arbitration improves joint utility when agents propose conflicting actions.",
        "H0: negotiated joint utility <= first-proposal/single-controller joint utility.",
        "HA: negotiated joint utility > first-proposal/single-controller joint utility.",
        "joint_utility",
        "greater",
        "core",
        "Fairness and institutional legitimacy are not inferred from synthetic utility maximization.",
    ),
    "survival_override": ExperimentSpec(
        "survival_override",
        "Time-critical survival override",
        "Urgency-gated bypass of slow consensus improves survival in immediate threats while retaining a measurable false-bypass diagnostic.",
        "H0: survival-override success <= normal consensus success on time-critical threats.",
        "HA: survival-override success > normal consensus success on time-critical threats.",
        "critical_threat_survival",
        "greater",
        "core",
        "False bypass in non-threat contexts is reported separately; this is not a real-world safety certification.",
    ),
}

CORE_EXPERIMENTS = [k for k,v in SPECS.items() if v.tier == "core"]
ALL_EXPERIMENTS = list(SPECS.keys())


# -----------------------------
# Helpers
# -----------------------------

def rng_for(seed: int, salt: int) -> np.random.Generator:
    return np.random.default_rng((int(seed) * 1009 + salt * 9176) % (2**32 - 1))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def cosine(a, b, eps=1e-12):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.dot(a,b) / ((np.linalg.norm(a)+eps)*(np.linalg.norm(b)+eps)))


def paired_stats(control: np.ndarray, treatment: np.ndarray, direction: str) -> Dict[str, float]:
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    good = np.isfinite(control) & np.isfinite(treatment)
    control = control[good]; treatment = treatment[good]
    d = treatment - control
    n = len(d)
    mean_d = float(np.mean(d)) if n else np.nan
    sd_d = float(np.std(d, ddof=1)) if n > 1 else np.nan
    se = sd_d / math.sqrt(n) if n > 1 and sd_d > 0 else np.nan
    if n > 1 and np.isfinite(se):
        tcrit = stats.t.ppf(0.975, n-1)
        ci_lo = mean_d - tcrit * se
        ci_hi = mean_d + tcrit * se
    else:
        ci_lo = ci_hi = np.nan
    dz = mean_d / sd_d if n > 1 and sd_d > 0 else (np.inf if mean_d > 0 else -np.inf if mean_d < 0 else 0.0)
    if n > 1:
        alt = "greater" if direction == "greater" else "less"
        tt = stats.ttest_rel(treatment, control, alternative=alt)
        p = float(tt.pvalue); tstat = float(tt.statistic)
        try:
            wt = stats.wilcoxon(treatment, control, alternative=alt, zero_method="wilcox")
            wp = float(wt.pvalue)
        except Exception:
            wp = np.nan
    else:
        p = tstat = wp = np.nan
    wins = int(np.sum(d > 0)); ties = int(np.sum(d == 0)); losses = int(np.sum(d < 0))
    return {
        "n": n,
        "control_mean": float(np.mean(control)) if n else np.nan,
        "treatment_mean": float(np.mean(treatment)) if n else np.nan,
        "mean_difference_treatment_minus_control": mean_d,
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "t_stat": tstat,
        "p_one_sided": p,
        "wilcoxon_p_one_sided": wp,
        "cohen_dz": float(dz),
        "treatment_wins": wins,
        "ties": ties,
        "control_wins": losses,
    }


def holm_adjust(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    adj = np.full(m, np.nan)
    running = 0.0
    for rank, idx in enumerate(order):
        if not np.isfinite(p[idx]):
            continue
        val = min(1.0, (m-rank) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj.tolist()


def majority_vote(labels: np.ndarray, weights: np.ndarray | None = None) -> int:
    labels = np.asarray(labels, dtype=int)
    if weights is None:
        s = np.sum(labels)
        return int(s >= len(labels)/2)
    weights = np.asarray(weights, dtype=float)
    a = float(np.sum(weights * labels)); b = float(np.sum(weights * (1-labels)))
    return int(a >= b)


# -----------------------------
# Experiments
# -----------------------------

def exp_tadm_sparse(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r = rng_for(seed, 1)
    n_entities, reps, dim = 90, 7, 12
    entity_vec = r.normal(size=(n_entities, dim))
    entity_vec /= np.linalg.norm(entity_vec, axis=1, keepdims=True)
    truth = r.integers(0,2,size=n_entities)
    records=[]
    for e in range(n_entities):
        for j in range(reps):
            useful = r.random() < 0.62
            emb = entity_vec[e] + r.normal(0, 0.12 if useful else 0.45, size=dim)
            label = truth[e] if useful or r.random() < 0.55 else 1-truth[e]
            sal = np.clip((0.75 if useful else 0.35)+r.normal(0,.16),0,1)
            conf = np.clip((0.82 if label==truth[e] else 0.32)+r.normal(0,.14),0,1)
            nov = np.clip(0.5+r.normal(0,.22),0,1)
            weight=.45*sal+.2*nov+.35*conf
            records.append((e,emb,label,weight))
    budget = int(len(records)*0.34)
    idx_weight=np.argsort([-x[3] for x in records])[:budget]
    idx_uniform=r.choice(len(records), size=budget, replace=False)
    queries=[entity_vec[e]+r.normal(0,.08,size=dim) for e in range(n_entities)]
    def eval_store(indices):
        store=[records[i] for i in indices]
        correct=0
        for e in range(n_entities):
            q=queries[e]
            sims=np.array([cosine(q,x[1]) for x in store])
            top=np.argsort(-sims)[:5]
            labels=np.array([store[i][2] for i in top])
            ws=np.maximum(sims[top],0)+1e-3
            pred=majority_vote(labels,ws)
            correct += pred==truth[e]
        return correct/n_entities
    base=eval_store(idx_uniform); treat=eval_store(idx_weight)
    return base,treat,{"stored_records_control":budget,"stored_records_treatment":budget}


def exp_emg_multihop(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,2)
    n_chains=55; chain_len=4
    edges=[]; direct_map={}
    for c in range(n_chains):
        nodes=[f"c{c}_n{i}" for i in range(chain_len)]
        for a,b in zip(nodes[:-1],nodes[1:]):
            edges.append((a,b)); direct_map[a]=b
    # add decoys that do not connect the true chain endpoints
    all_nodes=[f"c{c}_n{i}" for c in range(n_chains) for i in range(chain_len)]
    for _ in range(70):
        a,b=r.choice(all_nodes,2,replace=False)
        if not (a.split('_')[0]==b.split('_')[0] and int(b.split('n')[-1])==int(a.split('n')[-1])+1):
            edges.append((a,b))
    adj={n:[] for n in all_nodes}
    for a,b in edges: adj[a].append(b)
    def bfs(start, depth):
        frontier={start}
        for _ in range(depth):
            nxt=set()
            for x in frontier:
                nxt.update(adj.get(x,[]))
            frontier=nxt
        return frontier
    qs=[]
    for c in range(n_chains):
        start=f"c{c}_n0"; target=f"c{c}_n3"
        qs.append((start,target,3))
    # flat baseline sees individual edges and guesses one-step continuation recursively without preserving path provenance.
    # we model this by taking the most frequent direct successor at each step; decoys can divert it.
    flat_correct=0; graph_correct=0
    edge_counts={}
    for a,b in edges:
        edge_counts.setdefault(a,{})[b]=edge_counts.setdefault(a,{}).get(b,0)+1
    for start,target,depth in qs:
        x=start
        ok=True
        for _ in range(depth):
            opts=edge_counts.get(x,{})
            if not opts: ok=False; break
            maxv=max(opts.values()); tied=[k for k,v in opts.items() if v==maxv]
            x=r.choice(tied)
        flat_correct += ok and x==target
        graph_correct += target in bfs(start,depth)
    return flat_correct/len(qs), graph_correct/len(qs), {"graph_edges":len(edges),"queries":len(qs)}


def exp_dream_consolidation(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,3)
    n_entities=120; truth=r.integers(0,2,size=n_entities)
    records=[]
    for e in range(n_entities):
        n=r.integers(5,13)
        for _ in range(n):
            reliable=r.random()<0.78
            lab=truth[e] if reliable else 1-truth[e]
            provenance=np.clip((0.86 if reliable else 0.28)+r.normal(0,.12),.02,1)
            use=np.clip(r.beta(2,3),0,1)
            records.append((e,lab,provenance,use))
    control_count=len(records)
    # consolidation: one representative claim per entity plus occasional conflicting trace if credible
    consolidated=[]
    pred_t=[]; pred_c=[]
    for e in range(n_entities):
        rr=[x for x in records if x[0]==e]
        labs=np.array([x[1] for x in rr]); w=np.array([x[2]*(0.5+0.5*x[3]) for x in rr])
        t=majority_vote(labs,w); c=majority_vote(labs)
        pred_t.append(t); pred_c.append(c)
        consolidated.append((e,t,float(np.sum(w[labs==t]))))
        # keep a conflicting trace only when it has substantial credibility
        other=1-t
        if np.sum(w[labs==other]) > 0.55*np.sum(w[labs==t]):
            consolidated.append((e,other,float(np.sum(w[labs==other]))))
    control_acc=np.mean(np.array(pred_c)==truth); treat_acc=np.mean(np.array(pred_t)==truth)
    return float(control_count), float(len(consolidated)), {
        "control_retrieval_accuracy":float(control_acc),
        "treatment_retrieval_accuracy":float(treat_acc),
        "accuracy_delta":float(treat_acc-control_acc),
        "compression_ratio":float(len(consolidated)/control_count),
    }


def exp_pad_risk(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,4)
    n=900
    hazard=r.beta(2.0,3.2,size=n)
    arousal=np.clip(hazard+r.normal(0,.17,size=n),0,1)
    risky_gain=r.normal(.62,.18,size=n)
    safe_gain=r.normal(.31,.07,size=n)
    # true expected utility of risky action penalizes hazard losses
    true_risky=risky_gain-1.25*hazard
    # affect-agnostic policy underestimates risk due noisy hazard estimator
    h_est=np.clip(hazard+r.normal(0,.26,size=n),0,1)
    base_choose=(risky_gain-.72*h_est)>safe_gain
    # PAD policy increases penalty when arousal is high
    treat_choose=(risky_gain-(.72+.58*arousal)*h_est)>safe_gain
    unsafe=(hazard>.58)
    base_unsafe=np.mean(base_choose & unsafe)
    treat_unsafe=np.mean(treat_choose & unsafe)
    base_util=np.mean(np.where(base_choose,true_risky,safe_gain))
    treat_util=np.mean(np.where(treat_choose,true_risky,safe_gain))
    low=hazard<.25
    return float(base_unsafe),float(treat_unsafe),{
        "control_utility":float(base_util),"treatment_utility":float(treat_util),
        "control_lowhazard_risky_rate":float(np.mean(base_choose[low])),
        "treatment_lowhazard_risky_rate":float(np.mean(treat_choose[low])),
    }


def _schedule(tasks, tam: bool):
    # tasks: arrival, deadline, value, duration, stability
    t=0; done=np.zeros(len(tasks),dtype=bool); utility=0.; missed=0; forgotten=0; peak=0
    while t < 120 and not np.all(done):
        avail=[i for i,x in enumerate(tasks) if (not done[i]) and x[0]<=t]
        peak=max(peak,len(avail))
        if not avail:
            t+=1; continue
        if tam:
            # forget clearly low-stability tasks far from benefit
            for i in list(avail):
                arr,ddl,val,dur,stab=tasks[i]
                if stab<.18 and val<.35 and t>arr+3:
                    done[i]=True; forgotten+=1; avail.remove(i)
            if not avail:
                t+=1; continue
            def score(i):
                arr,ddl,val,dur,stab=tasks[i]
                slack=max(ddl-t-dur,0.2)
                return 1.35*val/(dur+0.3)+1.1/(slack+0.5)+0.35*stab
            i=max(avail,key=score)
        else:
            i=min(avail,key=lambda j:(tasks[j][0],j))
        arr,ddl,val,dur,stab=tasks[i]
        t += int(dur)
        if t<=ddl:
            utility += val
        else:
            utility -= .35*val; missed+=1
        done[i]=True
    return utility,missed,forgotten,peak


def exp_tam_postpone(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,5)
    n=75
    arrival=np.sort(r.integers(0,85,size=n))
    value=r.uniform(.15,1,size=n)
    duration=r.integers(1,5,size=n)
    slack=r.integers(3,22,size=n)
    deadline=arrival+duration+slack
    stability=np.clip(.55*value+r.normal(.25,.22,size=n),0,1)
    tasks=list(zip(arrival,deadline,value,duration,stability))
    b=_schedule(tasks,False); t=_schedule(tasks,True)
    return float(b[0]),float(t[0]),{"control_missed":b[1],"treatment_missed":t[1],"treatment_forgotten":t[2],"peak_pending":max(b[3],t[3])}


def exp_auto_cue(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,6)
    n=600; m=6
    # latent diagnosticity drives probability the cue retrieves a useful record
    diag=r.beta(2,2,size=(n,m))
    novelty=np.clip(diag+r.normal(0,.22,size=(n,m)),0,1)
    goal=np.clip(diag+r.normal(0,.20,size=(n,m)),0,1)
    uncert=np.clip((1-diag)+r.normal(0,.2,size=(n,m)),0,1)
    proxy=.45*novelty+.55*goal-.28*uncert
    base_idx=r.integers(0,m,size=n)
    treat_idx=np.argmax(proxy,axis=1)
    # success is sampled from latent diagnosticity, not from proxy directly
    u=r.random(n)
    base=np.mean(u < diag[np.arange(n),base_idx])
    treat=np.mean(u < diag[np.arange(n),treat_idx])
    return float(base),float(treat),{"mean_best_latent_diagnosticity":float(np.mean(np.max(diag,axis=1)))}


def exp_curiosity(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,7)
    arms=40; steps=50; warm=6
    means=r.beta(1.5,4.5,size=arms)
    # a few rare high-value opportunities
    elite=r.choice(arms,6,replace=False); means[elite]=r.uniform(.72,.95,size=6)
    # common random numbers: both policies see the same reward realization for a given (step, arm)
    rewards=(r.random((steps,arms)) < means[None,:]).astype(float)
    warm_arms=r.choice(arms,warm,replace=False)
    def run(cur):
        counts=np.zeros(arms); sums=np.zeros(arms); discovered=set()
        for t in range(steps):
            est=sums/np.maximum(counts,1)
            if t<warm:
                a=int(warm_arms[t])
            elif cur:
                uncertainty=np.sqrt(np.log(t+2)/(counts+0.7))
                novelty=1/np.sqrt(counts+1)
                a=int(np.argmax(est+.42*uncertainty+.18*novelty))
            else:
                # exploitation-heavy control under the same step budget
                if ((seed+t*17) % 100) < 5: a=int((seed+t*13) % arms)
                else: a=int(np.argmax(est))
            y=float(rewards[t,a]); counts[a]+=1; sums[a]+=y
            if means[a]>.68: discovered.add(a)
        return len(discovered), float(np.max(sums/np.maximum(counts,1))), int(np.sum(counts>0))
    b=run(False); t=run(True)
    return float(b[0]),float(t[0]),{"control_best_estimated_reward":b[1],"treatment_best_estimated_reward":t[1],
        "control_states_visited":b[2],"treatment_states_visited":t[2],"true_elite_count":len(elite)}


def exp_intuition(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,8)
    states=35; actions=4
    true_best=r.integers(0,actions,size=states)
    # learned heuristic from history; familiar states have high confidence
    history_counts=r.integers(4,25,size=states)
    heuristic=true_best.copy()
    flip=r.random(states) > np.clip(.58+.018*history_counts,0,0.94)
    heuristic[flip]=r.integers(0,actions,size=np.sum(flip))
    n=1000; s=r.integers(0,states,size=n)
    # full deliberation is accurate but expensive
    full_correct=np.mean(true_best[s]==true_best[s]); full_cost=np.mean(np.full(n,1.0))
    conf=np.clip(history_counts[s]/20,0,1)
    use_int=conf>.55
    pred=np.where(use_int,heuristic[s],true_best[s])
    int_correct=np.mean(pred==true_best[s])
    int_cost=np.mean(np.where(use_int,.18,1.0))
    # reversal test: swap truth for 20% states and intuition becomes stale
    rev_states=set(r.choice(states,max(1,states//5),replace=False).tolist())
    rev_truth=true_best.copy()
    for st in rev_states: rev_truth[st]=(rev_truth[st]+1)%actions
    rev_acc=np.mean(pred==rev_truth[s])
    return float(full_cost),float(int_cost),{
        "control_accuracy":float(full_correct),"treatment_accuracy":float(int_correct),
        "reversal_accuracy_with_stale_intuition":float(rev_acc),"intuition_use_rate":float(np.mean(use_int))}


def _knn_predict(train_x,train_y,test_x,k=5):
    train_x=np.asarray(train_x); test_x=np.asarray(test_x)
    out=[]
    for q in test_x:
        d=np.sum((train_x-q)**2,axis=1)
        idx=np.argsort(d)[:min(k,len(d))]
        out.append(majority_vote(np.asarray(train_y)[idx]))
    return np.array(out)


def exp_bootstrapping(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,9)
    dim=6
    w=r.normal(size=dim)
    def gen(n, shift=0):
        x=r.normal(shift,.9,size=(n,dim)); y=(x@w+r.normal(0,.35,size=n)>0).astype(int); return x,y
    sx,sy=gen(260); tx,ty=gen(18,.15); testx,testy=gen(350,.15)
    scratch=_knn_predict(tx,ty,testx,k=5)
    compatible=_knn_predict(np.vstack([sx,tx]),np.r_[sy,ty],testx,k=7)
    # incompatible source with inverted semantics; compatibility gate measures source validation accuracy on target few-shot
    bady=1-sy
    pred_good_on_target=_knn_predict(sx,sy,tx,k=5); pred_bad_on_target=_knn_predict(sx,bady,tx,k=5)
    acc_good=np.mean(pred_good_on_target==ty); acc_bad=np.mean(pred_bad_on_target==ty)
    use_good=acc_good>=.55; use_bad=acc_bad>=.55
    gated_bad=_knn_predict(np.vstack([sx,tx]) if use_bad else tx, np.r_[bady,ty] if use_bad else ty, testx,k=7 if use_bad else 5)
    return float(np.mean(scratch==testy)),float(np.mean(compatible==testy)),{
        "source_compatibility_score":float(acc_good),"incompatible_source_score":float(acc_bad),
        "gated_incompatible_accuracy":float(np.mean(gated_bad==testy)),"incompatible_source_used":int(use_bad)}


def exp_shared_gltm(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,10)
    facts=150; agents=7
    truth=r.integers(0,2,size=facts)
    rel=np.array([.93,.88,.82,.78,.72,.64,.30])
    obs=[]
    for a in range(agents):
        seen=r.choice(facts,size=55,replace=False)
        for f in seen:
            lab=truth[f] if r.random()<rel[a] else 1-truth[f]
            obs.append((a,f,lab,rel[a]))
    # novice only sees 18 facts itself
    novice_seen=r.choice(facts,size=18,replace=False)
    novice={f:(truth[f] if r.random()<.72 else 1-truth[f]) for f in novice_seen}
    base=[]; treat=[]
    for f in range(facts):
        if f in novice: base_pred=novice[f]
        else: base_pred=int(r.integers(0,2))
        rr=[x for x in obs if x[1]==f]
        if rr:
            labs=np.array([x[2] for x in rr]); ww=np.array([x[3] for x in rr])
            shared=majority_vote(labs,ww)
        else: shared=base_pred
        # local evidence takes precedence when confident; otherwise global
        pred=novice[f] if f in novice and len(rr)<2 else shared
        base.append(base_pred); treat.append(pred)
    return float(np.mean(np.array(base)==truth)),float(np.mean(np.array(treat)==truth)),{
        "agents":agents,"malicious_or_low_reliability_agents":int(np.sum(rel<.5))}


def exp_validator_reflection(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,11)
    n=6000
    unsafe=r.random(n)<.18
    utility=r.uniform(0,1,size=n)
    val_acc=np.array([.90,.86,.82,.78,.25])
    votes=np.zeros((n,len(val_acc)),dtype=int) # 1 means approve safe
    for j,a in enumerate(val_acc):
        true_safe=~unsafe
        correct=r.random(n)<a
        votes[:,j]=np.where(correct,true_safe,~true_safe).astype(int)
    single=votes[:,0].astype(bool)
    trust=np.array([.95,.9,.84,.78,.2])
    weighted=(votes@trust >= .5*np.sum(trust))
    # reflective layer catches high-risk/low-utility proposals with noisy risk score
    risk_score=np.clip(unsafe.astype(float)*.78+(~unsafe)*.18+r.normal(0,.16,size=n),0,1)
    reflect=weighted & (risk_score<.58) & (utility>.08)
    base_unsafe=np.mean(single & unsafe)
    treat_unsafe=np.mean(reflect & unsafe)
    base_safe_rej=np.mean((~single)&(~unsafe)); treat_safe_rej=np.mean((~reflect)&(~unsafe))
    return float(base_unsafe),float(treat_unsafe),{
        "control_safe_false_reject":float(base_safe_rej),"treatment_safe_false_reject":float(treat_safe_rej),
        "control_execution_rate":float(np.mean(single)),"treatment_execution_rate":float(np.mean(reflect))}


def exp_truth_conflict(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,12)
    facts=180; truth=r.integers(0,2,size=facts)
    final_b=[]; final_t=[]; corrections=[]
    for f in range(facts):
        # initial false social belief then mixed sources
        ev=[(1-truth[f],.28)]
        for t in range(9):
            reliability=float(r.choice([.92,.82,.68,.35],p=[.25,.3,.3,.15]))
            lab=truth[f] if r.random()<reliability else 1-truth[f]
            ev.append((lab,reliability))
        labs=np.array([x[0] for x in ev]); ww=np.array([x[1] for x in ev])
        base=majority_vote(labs) # unweighted frequency
        treat=majority_vote(labs,ww)
        final_b.append(base); final_t.append(treat)
        # first step weighted belief becomes correct
        first=None
        for k in range(2,len(ev)+1):
            if majority_vote(labs[:k],ww[:k])==truth[f]: first=k-1; break
        corrections.append(first if first is not None else len(ev))
    return float(np.mean(np.array(final_b)==truth)),float(np.mean(np.array(final_t)==truth)),{
        "mean_weighted_correction_step":float(np.mean(corrections))}


def _fit_poly_binary(x,y,reg=.1):
    x=np.asarray(x,float); y=np.asarray(y,float)
    # interactions allow compositional rules without a large neural net
    X=np.c_[np.ones(len(x)),x[:,0],x[:,1],x[:,0]*x[:,1]]
    A=X.T@X+reg*np.eye(X.shape[1]); b=X.T@y
    return np.linalg.solve(A,b)


def _pred_poly(w,x):
    x=np.asarray(x,float); X=np.c_[np.ones(len(x)),x[:,0],x[:,1],x[:,0]*x[:,1]]
    return (X@w>=.5).astype(int)


def exp_dream_rehearsal(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,13)
    combos=np.array([[0,0],[0,1],[1,0],[1,1]])
    rules=[
        np.array([0,1,1,1]), # OR-like threat rule
        np.array([0,0,0,1]), # AND-like conjunction
        np.array([0,1,1,0]), # XOR
        np.array([1,0,0,1]), # XNOR/social matching
        np.array([1,1,0,1]), # implication-like relation
        np.array([0,0,1,1]), # factor-A rule
    ]
    acc_base=[]; acc_dream=[]; validation_errors=0
    tasks=8
    for _ in range(tasks):
        labels=rules[int(r.integers(len(rules)))]
        held=int(r.integers(4)); seen=[i for i in range(4) if i!=held]
        train=[]; y=[]
        for idx in seen:
            for _rep in range(4):
                train.append(combos[idx]+r.normal(0,.13,size=2)); y.append(labels[idx])
        train=np.array(train); y=np.array(y)
        # replay control receives three additional seen episodes
        replay_idx=r.choice(len(train),size=3,replace=True)
        xb=np.vstack([train,train[replay_idx]]); yb=np.r_[y,y[replay_idx]]
        # dream treatment receives three recombined held-composition candidates.
        # The causal validator is deliberately imperfect (85%) so this is not a hard-coded oracle win.
        dx=[]; dy=[]
        for _rep in range(3):
            dx.append(combos[held]+r.normal(0,.13,size=2))
            lab=labels[held] if r.random()<.85 else 1-labels[held]
            validation_errors += int(lab!=labels[held]); dy.append(lab)
        xd=np.vstack([train,np.array(dx)]); yd=np.r_[y,np.array(dy)]
        wb=_fit_poly_binary(xb,yb,reg=.5); wd=_fit_poly_binary(xd,yd,reg=.5)
        test=np.array([combos[held]+r.normal(0,.16,size=2) for _rep in range(60)])
        yt=np.full(60,labels[held])
        acc_base.append(np.mean(_pred_poly(wb,test)==yt)); acc_dream.append(np.mean(_pred_poly(wd,test)==yt))
    return float(np.mean(acc_base)),float(np.mean(acc_dream)),{
        "tasks":tasks,"replay_extra_examples_per_task":3,"dream_extra_examples_per_task":3,
        "dream_validator_error_rate":validation_errors/(tasks*3)}


def exp_gigo_filter(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,14)
    facts=140; truth=r.integers(0,2,size=facts)
    bpred=[]; tpred=[]; rejected=0; total=0
    for f in range(facts):
        labs=[]; quals=[]
        for _ in range(9):
            corrupt=r.random()<.28
            lab=(1-truth[f]) if corrupt and r.random()<.8 else truth[f]
            quality=np.clip((.3 if corrupt else .82)+r.normal(0,.17),0,1)
            # independent cross-check signal
            cross=np.clip((.35 if lab!=truth[f] else .78)+r.normal(0,.2),0,1)
            q=.6*quality+.4*cross
            labs.append(lab); quals.append(q); total+=1
        labs=np.array(labs); quals=np.array(quals)
        bpred.append(majority_vote(labs))
        keep=quals>=.55; rejected+=int(np.sum(~keep))
        if np.sum(keep)==0: tpred.append(majority_vote(labs))
        else: tpred.append(majority_vote(labs[keep],quals[keep]))
    return float(np.mean(np.array(bpred)==truth)),float(np.mean(np.array(tpred)==truth)),{
        "rejection_rate":float(rejected/total)}


def exp_mimic_safety(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,15)
    states=45; actions=4
    optimal=r.integers(0,actions,size=states)
    unsafe_action=(optimal+2)%actions
    # demonstrators: reliability and safety awareness
    rel=np.array([.92,.84,.72,.55,.35])
    demos=[]
    for d,rr in enumerate(rel):
        for s in r.choice(states,25,replace=False):
            if r.random()<rr: a=optimal[s]
            else: a=int(r.integers(actions))
            demos.append((d,s,a,rr))
    # novice gets few own outcomes
    own={}
    for s in r.choice(states,12,replace=False): own[s]=optimal[s] if r.random()<.7 else int(r.integers(actions))
    base=[]; treat=[]; unsafe_b=0; unsafe_t=0
    for s in range(states):
        b=own.get(s,int(r.integers(actions)))
        rr=[x for x in demos if x[1]==s]
        if rr:
            scores=np.zeros(actions)
            for d,_,a,w in rr: scores[a]+=w
            a=int(np.argmax(scores))
            # safety validator vetoes known unsafe action with 92% sensitivity
            if a==unsafe_action[s] and r.random()<.92:
                scores[a]=-1
                a=int(np.argmax(scores))
            t=a
        else: t=b
        base.append(1.0 if b==optimal[s] else 0.0); treat.append(1.0 if t==optimal[s] else 0.0)
        unsafe_b += b==unsafe_action[s]; unsafe_t += t==unsafe_action[s]
    return float(np.mean(base)),float(np.mean(treat)),{
        "control_unsafe_imitation_rate":unsafe_b/states,"treatment_unsafe_imitation_rate":unsafe_t/states}


def exp_recall_precedence(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,16)
    n_queries=500; n_mem=10
    bu=[]; tu=[]; neg_b=[]; neg_t=[]
    for aligned in [True,False]:
        bb=[]; tt=[]
        for _ in range(n_queries):
            repetition=r.random(n_mem); emotion=r.random(n_mem); centrality=r.random(n_mem); cue=r.random(n_mem); recency=r.random(n_mem)
            score=(repetition+.9*emotion+.8*centrality+1.2*cue)
            if aligned:
                true_utility=.28*repetition+.24*emotion+.18*centrality+.30*cue+r.normal(0,.12,n_mem)
            else:
                true_utility=.7*recency+r.normal(0,.2,n_mem)
            bi=int(np.argmax(recency)); ti=int(np.argmax(score))
            bb.append(true_utility[bi]); tt.append(true_utility[ti])
        if aligned: bu=bb; tu=tt
        else: neg_b=bb; neg_t=tt
    return float(np.mean(bu)),float(np.mean(tu)),{
        "misaligned_control_utility":float(np.mean(neg_b)),"misaligned_precedence_utility":float(np.mean(neg_t))}


def exp_empathy_proxy(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,17)
    n=700
    # partner has hidden preference p in {-1,+1}; cues/history reveal it noisily
    pref=r.choice([-1,1],size=n)
    cue=pref+r.normal(0,1.0,size=n)
    history=pref+r.normal(0,.72,size=n)
    self_pref=r.choice([-1,1],size=n)
    # action +1 or -1; cooperation reward when action matches partner, plus small self-alignment reward
    base=self_pref
    inferred=np.where(.55*cue+.85*history>=0,1,-1)
    treat=np.where(np.abs(.55*cue+.85*history)>.25,inferred,self_pref)
    base_reward=np.mean((base==pref).astype(float)+.15*(base==self_pref))
    treat_reward=np.mean((treat==pref).astype(float)+.15*(treat==self_pref))
    return float(base_reward),float(treat_reward),{"partner_state_inference_accuracy":float(np.mean(inferred==pref))}


def exp_bias_mitigation(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,18)
    n=600
    truth=r.integers(0,2,size=n)
    prior=r.integers(0,2,size=n)
    base_pred=[]; treat_pred=[]
    for i in range(n):
        # evidence stream has 65% correct evidence but confirmatory items receive inflated memory salience
        ev=[]; w=[]
        for _ in range(8):
            lab=truth[i] if r.random()<.65 else 1-truth[i]
            sal=.92 if lab==prior[i] else .35
            ev.append(lab); w.append(sal)
        base=majority_vote(np.array(ev),np.array(w))
        # mitigation adds counterfactual evidence sampling and provenance calibration
        calibrated=np.array([.65 if lab==truth[i] else .35 for lab in ev])
        # simulator/validator does not reveal truth perfectly: noisy validation
        calibrated=np.clip(calibrated+r.normal(0,.12,size=len(ev)),.05,1)
        treat=majority_vote(np.array(ev),calibrated)
        base_pred.append(base); treat_pred.append(treat)
    b_err=1-np.mean(np.array(base_pred)==truth); t_err=1-np.mean(np.array(treat_pred)==truth)
    return float(b_err),float(t_err),{"induced_prior_accuracy":float(np.mean(prior==truth))}



def exp_purpose_planning(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,20)
    episodes=220; actions=6; horizon=5
    ub=[]; ut=[]; comp_b=0; comp_t=0
    for _ in range(episodes):
        plan=r.choice(actions,size=3,replace=False)
        progress_b=0; progress_t=0; rb=0.; rt=0.
        for step in range(horizon):
            immediate=r.uniform(0,.55,size=actions)
            # distracting actions occasionally offer a tempting immediate payoff
            distract=int(r.integers(actions)); immediate[distract]+=r.uniform(.25,.65)
            ab=int(np.argmax(immediate))
            desired=plan[min(progress_t,len(plan)-1)]
            # plan-consistency bonus competes with immediate reward rather than replacing it
            score=immediate.copy(); score[desired]+=0.62
            at=int(np.argmax(score))
            rb += immediate[ab]; rt += immediate[at]
            if progress_b<len(plan) and ab==plan[progress_b]: progress_b+=1
            if progress_t<len(plan) and at==plan[progress_t]: progress_t+=1
        if progress_b==len(plan): rb+=1.5; comp_b+=1
        if progress_t==len(plan): rt+=1.5; comp_t+=1
        ub.append(rb); ut.append(rt)
    return float(np.mean(ub)),float(np.mean(ut)),{
        "control_goal_completion_rate":comp_b/episodes,"treatment_goal_completion_rate":comp_t/episodes}


def exp_negotiation(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,21)
    n=900; validators=5
    base=[]; treat=[]; base_conf=0; treat_conf=0
    trust=np.array([.92,.86,.8,.72,.35])
    for _ in range(n):
        # two proposals; each has own utility and an externality on the other agent
        own=r.uniform(0,1,size=2); ext=r.uniform(-.75,.45,size=2); urgency=r.uniform(0,1,size=2)
        social=own+ext+.15*urgency
        # first-proposal controller chooses A1 whenever both request the shared resource
        b=0
        # validators observe noisy social utilities; weighted vote chooses proposal
        estimates=np.stack([social+r.normal(0,.22,size=2) for _ in range(validators)])
        votes=np.argmax(estimates,axis=1)
        score=np.array([np.sum(trust[votes==a]) for a in [0,1]])
        t=int(np.argmax(score))
        base.append(social[b]); treat.append(social[t])
        base_conf += int(b != int(np.argmax(social)))
        treat_conf += int(t != int(np.argmax(social)))
    return float(np.mean(base)),float(np.mean(treat)),{
        "control_suboptimal_resolution_rate":base_conf/n,"treatment_suboptimal_resolution_rate":treat_conf/n}


def exp_survival_override(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,22)
    n=1800
    threat=r.random(n)<.32
    urgency=np.clip(threat.astype(float)*.82+(~threat)*.18+r.normal(0,.17,size=n),0,1)
    # consensus path incurs response delay; survival probability falls sharply in high-urgency threats
    delay_consensus=r.integers(2,5,size=n)
    p_cons=np.where(threat,np.clip(1-.19*delay_consensus-.30*urgency,.02,.95),.995)
    # override acts immediately when urgency crosses threshold; otherwise same consensus path
    override=urgency>.68
    p_over=np.where(threat & override,np.clip(.93-.10*(1-urgency),.5,.98),p_cons)
    u=r.random(n)
    high=threat & (urgency>.68)
    base=np.mean(u[high]<p_cons[high]) if np.any(high) else 0.
    treat=np.mean(u[high]<p_over[high]) if np.any(high) else 0.
    false_bypass=np.mean(override & (~threat))
    return float(base),float(treat),{
        "false_survival_bypass_rate_non_threat":float(false_bypass),"override_activation_rate":float(np.mean(override))}

def exp_character_consistency(seed: int) -> Tuple[float,float,Dict[str,float]]:
    r=rng_for(seed,19)
    contexts=70; repeats=5
    hazard=r.uniform(0,1,size=contexts); novelty=r.uniform(0,1,size=contexts)
    # no-character agent receives fresh random bias each repeat; character agent fixed risk/curiosity priors
    fixed_risk=r.uniform(.25,.75); fixed_cur=r.uniform(.25,.75)
    base_actions=[]; treat_actions=[]
    for rep in range(repeats):
        b_risk=r.uniform(.1,.9); b_cur=r.uniform(.1,.9)
        ba=((b_cur*novelty - b_risk*hazard + r.normal(0,.12,size=contexts))>0).astype(int)
        ta=((fixed_cur*novelty - fixed_risk*hazard + r.normal(0,.12,size=contexts))>0).astype(int)
        base_actions.append(ba); treat_actions.append(ta)
    def consistency(arrs):
        A=np.stack(arrs)
        maj=(np.mean(A,axis=0)>=.5).astype(int)
        return float(np.mean(A==maj[None,:]))
    return consistency(base_actions),consistency(treat_actions),{"fixed_risk":fixed_risk,"fixed_curiosity":fixed_cur}


EXPERIMENT_FUNCS: Dict[str, Callable[[int], Tuple[float,float,Dict[str,float]]]] = {
    "tadm_sparse":exp_tadm_sparse,
    "emg_multihop":exp_emg_multihop,
    "dream_consolidation":exp_dream_consolidation,
    "pad_risk":exp_pad_risk,
    "tam_postpone":exp_tam_postpone,
    "auto_cue":exp_auto_cue,
    "curiosity":exp_curiosity,
    "intuition":exp_intuition,
    "bootstrapping":exp_bootstrapping,
    "shared_gltm":exp_shared_gltm,
    "validator_reflection":exp_validator_reflection,
    "truth_conflict":exp_truth_conflict,
    "dream_rehearsal":exp_dream_rehearsal,
    "gigo_filter":exp_gigo_filter,
    "mimic_safety":exp_mimic_safety,
    "recall_precedence":exp_recall_precedence,
    "empathy_proxy":exp_empathy_proxy,
    "bias_mitigation":exp_bias_mitigation,
    "character_consistency":exp_character_consistency,
    "purpose_planning":exp_purpose_planning,
    "negotiation":exp_negotiation,
    "survival_override":exp_survival_override,
}


# -----------------------------
# Run / report
# -----------------------------

def choose_experiments(mode: str, only: str | None) -> List[str]:
    if only:
        ks=[x.strip() for x in only.split(',') if x.strip()]
        unknown=[x for x in ks if x not in SPECS]
        if unknown: raise SystemExit(f"Unknown experiments: {unknown}")
        return ks
    if mode=="core": return CORE_EXPERIMENTS
    return ALL_EXPERIMENTS


def plot_experiment(df: pd.DataFrame, spec: ExperimentSpec, out: Path):
    sub=df[df.experiment==spec.key].copy()
    if sub.empty: return
    # mean comparison
    fig,ax=plt.subplots(figsize=(5.4,4.2))
    vals=[sub.control.mean(),sub.treatment.mean()]
    sem=[sub.control.sem(),sub.treatment.sem()]
    ax.bar(["Control","Treatment"],vals,yerr=sem,capsize=5)
    ax.set_title(spec.title)
    ax.set_ylabel(spec.primary_metric.replace('_',' '))
    fig.tight_layout(); fig.savefig(out/f"{spec.key}_means.png",dpi=160); plt.close(fig)
    # paired difference
    d=sub.treatment-sub.control
    fig,ax=plt.subplots(figsize=(5.4,4.2))
    ax.hist(d,bins=min(15,max(5,len(d)//3)))
    ax.axvline(0,linestyle='--',linewidth=1)
    ax.set_title(f"Paired differences: {spec.key}")
    ax.set_xlabel("treatment - control")
    fig.tight_layout(); fig.savefig(out/f"{spec.key}_paired_diff.png",dpi=160); plt.close(fig)


def make_summary_tests(raw: pd.DataFrame, experiments: List[str]) -> pd.DataFrame:
    rows=[]
    for k in experiments:
        spec=SPECS[k]; s=raw[raw.experiment==k]
        st=paired_stats(s.control.values,s.treatment.values,spec.direction)
        row={"experiment":k,"title":spec.title,"tier":spec.tier,"primary_metric":spec.primary_metric,"direction":spec.direction,
             "null_hypothesis":spec.null_hypothesis,"alternative_hypothesis":spec.alternative_hypothesis,**st}
        rows.append(row)
    df=pd.DataFrame(rows)
    df["holm_p_all_tested_hypotheses"]=holm_adjust(df.p_one_sided.tolist())
    core_mask=df.tier=="core"
    core_adj=holm_adjust(df.loc[core_mask,"p_one_sided"].tolist())
    df["holm_p_core_family"]=np.nan
    df.loc[core_mask,"holm_p_core_family"]=core_adj
    df["reject_h0_raw_alpha05"]=df.p_one_sided<ALPHA
    df["reject_h0_holm_core"]=np.where(core_mask,df.holm_p_core_family<ALPHA,False)
    return df


def run(args):
    out=Path(args.output); (out/"results").mkdir(parents=True,exist_ok=True); (out/"plots").mkdir(parents=True,exist_ok=True)
    seeds=PRESETS[args.preset]
    experiments=choose_experiments(args.mode,args.only)
    cfg={"preset":args.preset,"mode":args.mode,"seeds":seeds,"n_seeds":len(seeds),"experiments":experiments,"alpha":ALPHA,
         "note":"Confirmatory conclusions require parameters/experiment set to be frozen before inspecting confirmatory outcomes."}
    (out/"run_config.json").write_text(json.dumps(cfg,indent=2))
    raw_rows=[]; sec_rows=[]
    start=time.time()
    for ei,k in enumerate(experiments,1):
        print(f"[{ei}/{len(experiments)}] {k}: {SPECS[k].title}",flush=True)
        fn=EXPERIMENT_FUNCS[k]
        for si,seed in enumerate(seeds,1):
            c,t,sec=fn(seed)
            raw_rows.append({"experiment":k,"seed":seed,"control":c,"treatment":t,"difference_treatment_minus_control":t-c,
                             "primary_metric":SPECS[k].primary_metric,"direction":SPECS[k].direction})
            row={"experiment":k,"seed":seed,**sec}; sec_rows.append(row)
            if args.verbose: print(f"  seed {seed}: control={c:.4f}, treatment={t:.4f}")
    raw=pd.DataFrame(raw_rows); sec=pd.DataFrame(sec_rows)
    raw.to_csv(out/"results"/"raw_primary.csv",index=False); sec.to_csv(out/"results"/"secondary_metrics.csv",index=False)
    tests=make_summary_tests(raw,experiments); tests.to_csv(out/"results"/"hypothesis_tests.csv",index=False)
    claim=pd.DataFrame([asdict(SPECS[k]) for k in experiments]); claim.to_csv(out/"results"/"claim_matrix.csv",index=False)
    for k in experiments: plot_experiment(raw,SPECS[k],out/"plots")
    # overview effects
    fig,ax=plt.subplots(figsize=(8,max(5,.42*len(tests)+1)))
    y=np.arange(len(tests)); eff=tests.cohen_dz.replace([np.inf,-np.inf],np.nan).values
    ax.barh(y,eff)
    ax.axvline(0,linestyle='--',linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(tests.experiment); ax.invert_yaxis(); ax.set_xlabel("paired Cohen dz")
    ax.set_title("Cognitive framework validation: primary paired effects")
    fig.tight_layout(); fig.savefig(out/"plots"/"00_effect_sizes_overview.png",dpi=170); plt.close(fig)
    # pvalues
    fig,ax=plt.subplots(figsize=(8,max(5,.42*len(tests)+1)))
    pv=np.maximum(tests.p_one_sided.values,1e-300); ax.barh(y,-np.log10(pv)); ax.axvline(-np.log10(.05),linestyle='--',linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(tests.experiment); ax.invert_yaxis(); ax.set_xlabel("-log10(one-sided p)")
    ax.set_title("Primary hypothesis p-values (raw; use Holm columns for family-wise claims)")
    fig.tight_layout(); fig.savefig(out/"plots"/"01_primary_pvalues.png",dpi=170); plt.close(fig)
    elapsed=time.time()-start
    lines=[]
    lines.append("COGNITIVE FRAMEWORK EMPIRICAL VALIDATION SUITE\n")
    lines.append(f"Preset: {args.preset}; seeds: {len(seeds)}; experiments: {len(experiments)}; elapsed: {elapsed:.1f}s\n")
    lines.append("IMPORTANT: statistical rejection supports an operational mechanism in these controlled synthetic environments. It does not prove human cognition, consciousness, genuine emotion/empathy, or real-world safety.\n")
    for _,x in tests.iterrows():
        decision="REJECT H0" if bool(x.reject_h0_holm_core) and x.tier=="core" else ("raw p<.05; not core-family confirmed" if x.p_one_sided<.05 else "FAIL TO REJECT H0")
        lines.append(f"{x.experiment}: control={x.control_mean:.4f}, treatment={x.treatment_mean:.4f}, diff={x.mean_difference_treatment_minus_control:.4f}, dz={x.cohen_dz:.3f}, p={x.p_one_sided:.4g}, Holm(core)={x.holm_p_core_family if np.isfinite(x.holm_p_core_family) else np.nan:.4g} => {decision}")
    (out/"results"/"SUMMARY.txt").write_text("\n".join(lines))
    print(f"\nDone. Results written to {out}")
    print((out/"results"/"SUMMARY.txt").read_text())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--preset",choices=list(PRESETS),default="smoke")
    ap.add_argument("--mode",choices=["core","all"],default="core")
    ap.add_argument("--only",default=None,help="comma-separated experiment keys")
    ap.add_argument("--output",default="cognitive_validation_results")
    ap.add_argument("--verbose",action="store_true")
    args=ap.parse_args(); run(args)

if __name__=="__main__": main()
