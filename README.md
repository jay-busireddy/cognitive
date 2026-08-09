# Cognitive Framework Empirical Validation Suite

This package is a laptop-oriented mechanism-validation harness for the paper **A Cognitive-Inspired Unified Framework for Adaptive AI Systems**.

It is designed to answer a narrower and scientifically defensible question than "does the whole cognitive theory work?":

> When one mechanism is added or removed while the environment, random seed, task budget, and evaluation set are held fixed, does the prespecified behavior predicted by the paper appear on held-out synthetic tasks with known causal ground truth?

The suite contains **18 core confirmatory hypotheses** and **4 proxy/construct experiments**. It also includes a separate descriptive resource-scaling benchmark.

## Main script

`cognitive_validation.py`

Presets:

- `smoke`: 2 seeds, software verification only.
- `pilot`: 8 fresh seeds, engineering/debugging only. Do not use it as confirmatory evidence if you use its results to tune parameters.
- `confirmatory`: 60 fresh matched seeds. This is the inferential run.

Modes:

- `core`: 18 core hypotheses only.
- `all`: core + 4 proxy experiments (recall precedence, empathy proxy, bias mitigation, character consistency).

## Core experiments

1. TADM selective sparse memory under a fixed memory budget.
2. EMG multi-hop temporal/causal reasoning.
3. Dream consolidation for redundancy/conflict compression.
4. PAD-modulated risk-sensitive action selection.
5. Temporal Action Modulation (remember/postpone/forget).
6. Automatic cue determination.
7. Curiosity-guided novelty/uncertainty exploration.
8. Intuition heuristic priors.
9. Knowledge bootstrapping / memory transplant.
10. SharedGLTM global-memory transfer.
11. Validator consensus + reflective safety layer.
12. Incremental confidence/credibility conflict resolution.
13. Validated counterfactual dream rehearsal.
14. Input-quality gating / GIGO mitigation.
15. Trust-weighted observational imitation with safety validation.
16. Purpose-conditioned planning.
17. Multi-agent negotiation/arbitration.
18. Time-critical survival override.

## Proxy experiments

These are useful behavioral tests but must not be described as proof of human psychological states:

- Recall precedence from repetition/emotion/centrality/cue match.
- Social-state projection (`empathy_proxy`).
- Counterfactual mitigation of deliberately induced memory bias.
- Persistent character-prior consistency.

## Important interpretation rule

A statistically significant result supports the **operationalized mechanism in the controlled synthetic environment**. It does **not** by itself establish genuine emotion, empathy, consciousness, human-like character, ethical correctness, or real-world safety.

A non-significant result means **fail to reject the null**. It does not prove the null is true.

## Output

The run creates:

- `results/raw_primary.csv`
- `results/secondary_metrics.csv`
- `results/hypothesis_tests.csv`
- `results/claim_matrix.csv`
- `results/SUMMARY.txt`
- `run_config.json`
- 2 plots per experiment plus overview plots

The hypothesis table contains raw one-sided paired p-values, paired Cohen's dz, 95% CIs for the paired difference, Wilcoxon sensitivity p-values, and Holm-adjusted p-values. **Use the Holm-adjusted core-family column for the final family-wise confirmatory claims.**

## Separate resource benchmark

`resource_benchmark.py` measures actual local dense-vs-sparse representation bytes and query time across growing memory sizes. It is descriptive and hardware-specific, not part of the 18-hypothesis family.

