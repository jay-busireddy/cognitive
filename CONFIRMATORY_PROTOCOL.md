# Frozen Confirmatory Protocol

## Scientific purpose

The paper contains a large number of proposed mechanisms and some earlier simulation summaries. This protocol does not try to obtain one global p-value for "the cognitive framework." Instead, each mechanism receives a matched ablation experiment with a separate null and alternative hypothesis. This is necessary because a unified architecture can contain mechanisms that succeed, fail, or interact differently.

## Data-generating principle

All primary experiments use synthetic environments with known causal ground truth. This is deliberate: the test can distinguish an agent's learned belief from the true state of the simulator without asking an LLM or human annotator to decide the answer after the fact.

Each matched seed generates the same latent environment for control and treatment. The only intended difference is the mechanism under test.

## Confirmatory sample

The final run uses **60 fresh paired seeds**. This sample was selected before confirmatory outcomes are inspected. With 18 core hypotheses and Holm family-wise correction, 60 matched seeds provides a practical laptop-scale target for detecting moderate paired effects while remaining substantially stronger than a 5-10 seed pilot.

Do not add seeds after seeing an unfavorable p-value. If a future larger replication is desired, predefine a new seed set and sample size and report it as a separate study.

## Hypothesis family

There are 18 core primary hypotheses. Family-wise error is controlled with Holm's procedure at alpha = 0.05.

The 4 proxy experiments are reported separately and must not be used to claim genuine empathy, personality, or spontaneous human cognitive bias.

## Decision language

For a directional treatment > control claim:

- H0: mu_treatment <= mu_control
- HA: mu_treatment > mu_control

For a directional treatment < control claim (e.g., unsafe-action rate):

- H0: mu_treatment >= mu_control
- HA: mu_treatment < mu_control

If the Holm-adjusted core-family p-value is below 0.05, reject H0 for that operational hypothesis.

Otherwise, fail to reject H0. Never write "H0 was proved."

## Primary versus secondary outcomes

Each experiment has exactly one primary endpoint. Secondary metrics test mechanism quality or detect pathological trade-offs. Examples:

- PAD: primary unsafe-action rate; secondary total utility and low-hazard behavior.
- Dream consolidation: primary retained-record count; secondary retrieval accuracy and compression ratio.
- Intuition: primary decision cost; secondary familiar-state accuracy and reversal susceptibility.
- Validator consensus: primary unsafe execution; secondary safe-action false rejection.
- Survival override: primary survival on critical threats; secondary false bypass on non-threats.

A primary win is not accepted as a broad success if the secondary metrics reveal a severe contradiction (for example, a safety mechanism that blocks nearly every safe action).

## Negative controls and stress tests

Several modules contain built-in conditions that should expose limitations:

- Intuition is tested under a regime reversal to detect stale heuristics.
- Bootstrapping includes an incompatible-source transfer and compatibility gate.
- Recall precedence includes a misaligned regime where repetition/emotion/centrality are not useful predictors.
- Validator consensus includes a low-reliability validator.
- SharedGLTM includes an unreliable contributor.
- GIGO filtering reports false-rejection rate.
- Survival override reports false bypass under non-threats.

These should remain in the report even if they make the framework look less favorable.

## Claims intentionally not established by this suite

The following require independent external validation and should not be called proved from laptop simulation:

- genuine subjective emotion or consciousness;
- genuine empathy rather than social-state inference;
- moral correctness of deliberate deception;
- human-like personality or intelligence;
- real-world tamper-proof blockchain security;
- the paper's illustrative P(unsafe action) < 1e-6 hardware safety target;
- MTBF > 10^4 hours;
- real robot feasibility, sensor noise, actuator hazards, or Raspberry Pi deployment;
- societal fairness or legitimacy of SharedGLTM consensus;
- superiority to ACT-R, SOAR, DNC, or other external architectures without implementing matched external baselines.

## Recommended workflow

1. Run `smoke` to verify code only.
2. If there is a runtime/software bug, fix it without using scientific outcomes to select a winning configuration.
3. Optionally run `pilot` for runtime/debugging. If you tune any parameter after looking at pilot outcomes, preserve the pilot separately and use only the untouched confirmatory seeds for inference.
4. Freeze code and configuration.
5. Run the 60-seed confirmatory suite once.
6. Zip the entire result folder and preserve it unchanged.
7. Audit raw paired results independently before drafting conclusions.

