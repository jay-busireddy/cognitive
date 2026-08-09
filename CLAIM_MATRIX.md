# Claim-to-Experiment Matrix

This matrix maps the paper's main operational claims to the validation suite. It deliberately separates **mechanism evidence** from stronger psychological or real-world interpretations.

| Paper section / construct | Operational claim tested | Experiment key | Primary endpoint | Status class |
|---|---|---|---|---|
| 6.3.1 / 7.2 TADM | Salience/novelty/confidence-weighted sparse writes preserve more useful information under fixed memory | `tadm_sparse` | retrieval accuracy | core |
| 6.3.2 / 7.3 EMG | Structured edges support temporal/causal multi-hop retrieval better than flat episodic lookup | `emg_multihop` | multi-hop accuracy | core |
| 6.3.5 / 7.9 Dream consolidation | Merge/prune/conflict-resolution reduces redundant memory | `dream_consolidation` | retained records | core |
| 6.3.3 / 7.4 PAD | Affect/arousal-sensitive risk penalty reduces unsafe action in high-hazard contexts | `pad_risk` | unsafe-action rate | core |
| 6.3.4 / 7.7 / 8.4 TAM | Remember/postpone/forget improves deadline-weighted utility under finite attention | `tam_postpone` | deadline utility | core |
| 8.5 Auto-cue | Benefit-oriented cue selection improves retrieval/task success | `auto_cue` | cue success | core |
| 8.6 Curiosity | Novelty/uncertainty bonus improves useful discovery under equal interaction budget | `curiosity` | useful discoveries | core |
| 8.7 Intuition | Familiar-state heuristic prior reduces deliberation cost | `intuition` | decision cost | core |
| 6.4.3 / 13.2 Bootstrapping | Compatible memory transplant accelerates few-shot adaptation | `bootstrapping` | few-shot accuracy | core |
| 8.3 / 8.11 Shared global learning | Reliability-weighted SharedGLTM helps inexperienced agents | `shared_gltm` | novice accuracy | core |
| 6.6 / 6.7 Reflective + validator safety | Consensus plus reflection reduces unsafe execution | `validator_reflection` | unsafe-execution rate | core |
| 8.12 / 13.1.1-13.1.3 Conflict/truth | Credibility/provenance weighting corrects false beliefs and resists noise | `truth_conflict` | final fact accuracy | core |
| 7.10 / 8.1 Dream simulation and recombination | Validated counterfactual recombination improves unseen composition beyond replay | `dream_rehearsal` | composition accuracy | core |
| 10.6 / 13.1.4 GIGO | Quality/cross-validation gating reduces belief contamination | `gigo_filter` | belief accuracy | core |
| 6.4.6-6.4.7 Mimicking | Trust-weighted imitation improves few-shot skill acquisition with safety gating | `mimic_safety` | few-shot reward | core |
| 7.5 Purpose/planning | Goal-conditioned planning outperforms myopic action choice on multi-step objectives | `purpose_planning` | goal-completion utility | core |
| 6.5.3-6.5.5 Negotiation | Trust-weighted arbitration improves joint utility under conflicting proposals | `negotiation` | joint utility | core |
| 6.7.7 Survival override | Urgency-gated bypass improves survival under time-critical threat | `survival_override` | critical-threat survival | core |
| 13.3 Recall precedence | repetition + affect + centrality + cue-match predicts useful first recall when aligned with task utility | `recall_precedence` | first-recall utility | proxy |
| 8.2 Empathy | social-state projection improves coordination | `empathy_proxy` | coordination reward | proxy only; not genuine empathy |
| 13.7 Bias | counterfactual validation mitigates deliberately induced memory bias | `bias_mitigation` | bias error | proxy only |
| 8.8 / 13.4 Character | persistent priors create stable behavioral profiles | `character_consistency` | profile consistency | proxy only; not personality |
| 10.3 / 11.2 resource claims | sparse representations change memory footprint/query time with scale | `resource_benchmark.py` | bytes and local latency | descriptive implementation benchmark |

## Claims not adequately established by the laptop suite

The following should remain separate future studies rather than being forced into synthetic significance tests:

1. **Genuine emotion, empathy, self-awareness, metacognition, or consciousness.** The suite can test state variables and policy effects, not subjective experience.
2. **Ethical correctness of deliberate deception.** A synthetic utility function can verify policy consistency, but the moral label comes from externally chosen normative assumptions.
3. **Human-like intelligence or personality.** Behavioral consistency and transfer are measurable; equivalence to human psychological constructs is not.
4. **Real-world safety guarantees** such as `P(unsafe action) < 10^-6` or `MTBF > 10^4 hours`. These require orders of magnitude more trials and hardware-specific validation.
5. **Tamper-proof blockchain/security claims.** This requires a security/threat-model study, Byzantine/adversarial testing, and cryptographic analysis.
6. **Raspberry Pi / robotics feasibility.** The included resource benchmark is only a desktop/laptop implementation benchmark; actual sensor, actuator, thermal, latency, and power tests require hardware.
7. **Superiority to ACT-R, SOAR, NTM, DNC, or other architectures.** A fair external-baseline paper requires implementing those systems on matched tasks and budgets.
8. **Societal fairness or legitimacy of SharedGLTM.** Synthetic joint utility is not a substitute for governance or human-subject evaluation.

## Recommended publication structure after the run

A future empirical sister paper should report results by mechanism rather than claim one binary verdict for the whole framework:

- Memory and retrieval mechanisms
- Temporal control and planning
- Affective/risk modulation
- Curiosity/intuition/character
- Dream consolidation and counterfactual simulation
- Knowledge transfer and global memory
- Safety, reflection, negotiation, and survival override
- Truth/conflict/GIGO robustness
- Proxy cognitive-trait experiments
- Resource-scaling benchmark

A theory can therefore be **partially supported**: some nulls may be rejected while others fail to reject. This is more informative than tuning the unified system until one aggregate score improves.
