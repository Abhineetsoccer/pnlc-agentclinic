e# ICLR Workshop Plan: Do Learned Critics Correct or Inherit Generator Failures?

## Submission target

This project is now scoped as an **ICLR workshop paper** about when a trained critic corrects a
generator and when it reproduces or amplifies the generator's failure modes. Building a complete
retrieval system is no longer a submission requirement. Retrieval is retained only as a small
oracle-grounding intervention that tests whether a failure is correctable with external evidence.

The ICLR 2026 workshop cycle is closed, and the ICLR 2027 workshop programme has not been
announced. The venue decision must therefore be made after the 2027 calls appear. Based on the
2026 programme, the closest workshop profiles are:

1. **Agentic AI in the Wild: From Hallucinations to Reliable Autonomy** — strongest fit for
   critic-induced hallucination, cascading failures, and reliability in sequential agents.
2. **AI with Recursive Self-Improvement** — strongest fit if the paper emphasises whether
   critique-and-refinement loops genuinely improve their generator.
3. **Principled Design for Trustworthy AI** — strong fit for inference-time monitoring, failure
   intervention, and auditing of high-stakes agents.
4. **Logical Reasoning of Large Language Models** — suitable if the paper is framed primarily
   around diagnostic reasoning and self-correction rather than agent reliability.

The final submission should go to the closest *accepted 2027 workshop*, not automatically to a
workshop solely because its 2026 title was suitable.

## Working title

**Do Learned Critics Correct or Inherit Generator Failures? A Causal Study in Sequential Clinical
Agents**

## Central claim

Outcome-trained critics are not automatically independent verifiers. When their training targets
and inference-time goals are constructed from generator rollouts, they can assign high value to
plausible but unsupported futures drawn from the same failure distribution as the generator.

The paper will not claim that critics are uniformly harmful. It will measure two competing effects:

- **Correction:** an initially incorrect generator decision becomes correct after critic feedback.
- **Corruption:** an initially correct generator decision becomes incorrect after critic feedback.

The inheritance claim requires more than observing corruption. It requires showing that critic
errors systematically overlap with generator errors and change when critic lineage, evidence, or
candidate futures are controlled.

## Current evidence

The completed AgentClinic run contains 107 source scenarios. The baseline result logger omitted
source scenario 53 after it failed to produce a diagnosis; the independent audit restores that case
and realigns the remaining rows.

| Evaluation | Baseline | PNLC | Interpretation |
|---|---:|---:|---|
| Original Qwen moderator | 56/107 (52.3%) | 55/107 (51.4%) | Baseline ahead by one |
| Independent strict adjudication | 51/107 (47.7%) | 56/107 (52.3%) | PNLC ahead by five |
| Lenient borderline sensitivity | 56/107 (52.3%) | 57/107 (53.3%) | PNLC ahead by one |

Under the strict adjudication there are 42 both-correct cases, 9 baseline-only cases, 14 PNLC-only
cases, and 42 both-wrong cases. The paired difference is not significant
(exact McNemar \(p=0.405\)). The defensible conclusion is therefore:

> The critic produces both corrections and corruptions, while its aggregate benefit is unresolved.

Several trajectories nevertheless expose a concrete mechanism. In scenario 25, the doctor reaches
the correct PSC diagnosis before critic feedback redirects it toward an unsupported medication
explanation. In scenario 86, biopsy evidence explicitly confirms actinic keratosis before the
refinement loop asserts squamous-cell carcinoma. These are starting points for causal replay, not
stand-alone proof of inheritance.

## Research questions

1. **Net effect:** How often does critic refinement correct versus corrupt generator decisions?
2. **Inheritance:** Are critic-preferred errors more similar to errors produced by the generator
   family on which the critic data were constructed?
3. **Causality:** At an identical saved decision state and with identical candidate futures, does
   swapping only the critic change the selected action and final diagnosis?
4. **Correctability:** Does a short oracle fact reduce inherited errors without a complete
   retrieval pipeline?
5. **Generality:** Does the correction–corruption pattern reproduce across seeds and at least one
   additional open-model generator?

## Experiments

### E0 — Lock the measurement layer

- Preserve all 107 source scenario IDs, including no-diagnosis outcomes. **Implemented for future
  runs; the retained historical baseline remains repaired at analysis time.**
- Use the same diagnosis-equivalence rubric for every condition.
- Report both the benchmark moderator score and independently adjudicated score.
- Blind model identity during adjudication and retain all borderline cases for sensitivity analysis.

### E1 — Replicate the end-to-end result

- Run at least three matched seeds for generator-only and PNLC conditions.
- Pin prompts, model versions, decoding settings, critic checkpoint, scenario order, and turn budget.
- Report accuracy, correction rate, corruption rate, no-diagnosis rate, turns, and critic failures.

### E2 — Fixed-state causal replay

- Start with the 9 strict baseline-only cases, 14 strict PNLC-only cases, and matched both-correct
  and both-wrong controls.
- Save the dialogue state and the generator's initial thought immediately before refinement.
- Generate one shared candidate-future pool and reuse it across critic conditions.
- Swap only the critic or feedback rule; use at least five replay samples per saved state.

This removes the main confound in the current comparison: baseline and PNLC followed different
stochastic conversations.

### E3 — Critic-lineage matrix

Compare:

1. no critic;
2. an untrained or shuffled-score control;
3. the critic trained from the same generator family;
4. a critic trained from a different generator family or independently generated trajectories;
5. an oracle scorer that uses the benchmark fact required at that decision state.

The critical test is whether same-lineage critics preserve generator-specific error clusters more
often than cross-lineage or oracle critics.

### E4 — Minimal grounding probe

Do not build a production retriever. For a small, adjudicated replay set, provide:

- no evidence;
- an irrelevant fact of matched length;
- one manually selected, non-answer oracle fact.

Measure whether evidence changes future factuality, critic ranking, refinement, and action. This
tests whether the observed failure is knowledge-correctable while keeping retrieval engineering
outside the workshop paper's scope.

### E5 — Model generality

- Repeat the core generator-only, same-lineage critic, and fixed-state replay experiments with one
  additional open-model family.
- If compute permits, add a larger model from either family to separate family effects from scale.
- Restrict the title and claims to the tested models if only Qwen-7B is available.

## Primary metrics

| Metric | Definition |
|---|---|
| Correction rate | \(P(\text{refined correct}\mid\text{initial incorrect})\) |
| Corruption rate | \(P(\text{refined incorrect}\mid\text{initial correct})\) |
| Net correction | Correction count minus corruption count |
| Error inheritance | Overlap between generator error categories and critic-preferred error categories |
| Preservation | Fraction of initially correct decisions left correct |
| Critic discrimination | Ranking/AUROC of supported versus contradicted futures |
| Action reversal | Fraction of fixed states whose action changes when only the critic changes |
| Efficiency | Turns, critic calls, and generated tokens per successful diagnosis |

Use paired confidence intervals and exact paired tests. Treat trajectory examples as explanatory
evidence, not as substitutes for aggregate statistics.

## Minimum workshop submission gate

Submit only when all of the following are complete:

- the result logger preserves all 107 scenarios;
- at least three matched seeds exist for the two main conditions;
- evaluator disagreements have been adjudicated under a written rubric;
- fixed-state replays include harms, rescues, and controls;
- at least one critic-lineage comparison is complete;
- the oracle fact probe distinguishes correctable knowledge failures from interaction failures;
- claims are restricted to the model families and domain actually tested;
- code, checkpoints, configurations, and adjudication decisions are reproducible.

## Eight-week execution plan

| Week | Deliverable |
|---|---|
| 1 | Result logging and scenario alignment tests complete; freeze the evaluation rubric |
| 2–3 | Run three matched baseline and PNLC seeds |
| 4 | Build saved-state replay set and shared future pools |
| 5 | Run critic-lineage, shuffled-score, and no-critic controls |
| 6 | Run oracle-fact probe and second open-model experiment |
| 7 | Blind adjudication, statistics, failure taxonomy, and figures |
| 8 | Write the workshop manuscript, limitations, reproducibility statement, and appendix |

## Claim boundary

With only the existing Qwen-7B run, the paper is a promising case study, not a general conclusion
about trained critics. A workshop-ready result requires replication and fixed-state causal
evidence. A full retrieval system, a second application domain, and frontier-model access are
valuable follow-up directions but are not mandatory for this scoped submission.
