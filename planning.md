# Research Plan: Positional Bias in Multiple-Choice Benchmarks

Written before any model was run (pre-registration of the analysis). Deviations are logged at the bottom.

## Situation at start (affects design)

- The resource-finder stage was **skipped** (`.neurico/pipeline_state.json`: `"skipped": true`). There is no
  `literature_review.md`, `resources.md`, `papers/`, `datasets/` or `code/` in the workspace. The background
  below is therefore from my own knowledge of the literature, not from a gathered review.
- `OPENROUTER_KEY` (and every other LLM API key) is **not set** in the environment. I cannot call hosted models.
- Available instead: 4x NVIDIA RTX A6000 (48 GB each) and a local HuggingFace cache containing real
  open-weight instruct models and MMLU / CommonsenseQA.

Decision: run **real open-weight LLMs locally** and read their answer-token probabilities. This is not a
simulation, and it gives full option-probability distributions, which hosted APIs mostly do not. The cost is
that conclusions cover 1.5B-14B open models, not frontier API models.

## Motivation & Novelty Assessment

### Why This Research Matters
Multiple-choice (MC) accuracy is the default currency for comparing LLMs. If a model's choice depends on *where*
the correct answer sits, a benchmark score partly measures the alignment between the benchmark's answer-key
distribution and the model's positional preference rather than knowledge. Benchmark builders, leaderboard
maintainers and anyone comparing models with small accuracy gaps need to know how large that component is.

### Gap in Existing Work
Prior work (Zheng et al. 2024 "LLMs are not robust multiple choice selectors" / PriDe; Pezeshkpour & Hruschka 2023;
Robinson & Wingate 2023; Wang et al. 2024 on first-token evaluation) establishes that selection bias exists and
that it is partly a token (label) bias and partly a position bias. Less well covered: (i) how the effect scales
with the **number of options when question content is held fixed** — comparisons across benchmarks with
different k confound format with content; (ii) a **difficulty-stratified** estimate where difficulty is not
defined by the same responses used to measure the bias; (iii) translating the bias into "how many accuracy
points" under explicit, competing definitions of inflation.

### Our Novel Contribution
A within-question design: every question is shown with the gold answer at every position (cyclic rotations),
at k = 2, 3, 4 (MMLU) and k = 2, 3, 4, 5 (CommonsenseQA) built from the *same* questions, plus native
True/False. Difficulty is measured leave-one-model-out, so the stratification is not circular.

### Experiment Justification
- **E1 (4-option, MMLU)**: tests the core claim — early-position preference and a 3-8 point inflation.
- **E2 (option-count ladder, same questions)**: tests "T/F smallest (<2), 5-option largest (>10)" without
  content confound.
- **E3 (native True/False, BoolQ)**: the ladder's k=2 is still an A/B lettered MC item; a real true/false
  format is needed to speak to the hypothesis as worded.
- **E4 (difficulty stratification)**: tests "strongest at medium difficulty" and "up to 15% of apparent accuracy".
- **E5 (label ablation: A-D vs 1-4)**: separates preference for the *token* "A" from preference for the *slot*.
- **E6 (read-out validation)**: checks that first-token logits agree with what the model writes when it generates.

## Research Question
Do current instruct LLMs prefer early answer positions in MC questions, by how much does that change measured
accuracy, and how does the effect depend on option count and question difficulty?

## Hypothesis Decomposition
- **H1** Models over-select early positions (A/B) relative to the uniform rate implied by a balanced key.
- **H2** On 4-option items, positional bias inflates accuracy by 3-8 points.
- **H3** The positional effect is largest for medium-difficulty questions.
- **H4** True/False shows <2 points; 5-option shows >10 points.
- **H5** Near the decision boundary, up to 15% of apparent accuracy is attributable to position.

Independent variables: gold position, number of options k, label set, model, question difficulty.
Dependent variables: correctness, chosen position, option probabilities.

## Proposed Methodology

### Data
- MMLU test (4 options): stratified random sample of 1,500 questions across the 57 subjects.
- CommonsenseQA validation (5 options): all 1,221 questions.
- BoolQ validation (yes/no reading comprehension, shown as True/False statements-about-a-passage): 1,500 sampled.
- Reduced-k variants: keep the gold option plus (k-1) distractors sampled with a fixed seed.

### Presentation
For a k-option item, all k cyclic rotations of the option list are shown, so the gold answer occupies every
position exactly once per question (k prompts/question). Rotation 0 of the native-k item is the as-published order.
Zero-shot, each model's own chat template, one fixed instruction ("Answer with the letter only").

### Read-out
Next-token logits at the first assistant token, restricted to the k label tokens, softmaxed. Prediction = argmax.
Deterministic (no sampling), so there is no seed variance in model outputs; uncertainty comes from question sampling.

### Models
Open-weight instruct models released 2024 or later, from more than one family, limited to what is in the local
cache (disk has ~60 GB free, so no new large downloads): Qwen2.5 (1.5B, 7B, 14B — a size ladder within one
family), Llama-3.1-8B-Instruct, Phi-3-mini, Qwen3-family instruct model. Exact set fixed after a load test.

### Metrics (defined now, all reported)
Let acc_p = accuracy when gold is at position p; bal = mean_p acc_p (position-balanced accuracy).
1. **Selection share** of each position under the balanced key (expected 1/k); chi-square goodness of fit.
2. **Position spread** = max_p acc_p - min_p acc_p; **RStd** = std of acc_p (Zheng et al.).
3. **Inflation**, three definitions, because "inflates scores" is ambiguous:
   - (a) early-key inflation = mean(acc_A, acc_B) - bal — what a benchmark with all keys early would gain;
   - (b) best-position inflation = max_p acc_p - bal — upper bound for an adversarially aligned key;
   - (c) as-published inflation = acc(original order) - bal — what the real benchmark gains.
   H2 is judged primarily on (c), since that is the score people report, with (a) as the test of the
   "early positions" mechanism.
4. **Flip rate**: share of questions whose correctness is not constant across rotations.
5. **Position-attributable share** in a stratum = (max_p acc_p - bal) / max_p acc_p.

### Difficulty
Primary: leave-one-model-out difficulty = mean balanced correctness of the *other* models on that question,
cut into terciles/quintiles. Secondary (circular, reported with that caveat): the model's own rotation-averaged
probability of the gold option.

### Statistical Analysis Plan
- Question-level bootstrap (2,000 resamples, questions as clusters) for 95% CIs on every metric.
- Cochran's Q across gold positions per model x dataset; chi-square for selection shares.
- Paired bootstrap for differences between formats and between difficulty strata.
- Holm correction within each family of tests; alpha = 0.05.

## Expected Outcomes
Support for H2 needs (c) or (a) in [3, 8] points for most models; H4 needs the T/F effect < 2 and k=5 effect > 10
on the same metric; H3 needs the spread in the middle difficulty stratum to exceed both extremes with
non-overlapping paired CIs. Anything else is reported as not supported — including a preference for late
positions, which would contradict H1's direction.

## Timeline
Setup and data 30 min; harness and smoke test 45 min; full runs 60-90 min (4 GPUs in parallel);
analysis 45 min; report 30 min.

## Potential Challenges
- First-token read-out may differ from generated text -> E6.
- Cyclic rotation covers k of k! orders; gold position is balanced but distractor order is not fully randomized.
- MMLU/CSQA are almost certainly in training data; this affects absolute accuracy, less so order sensitivity,
  but memorised original orders could themselves favour rotation 0. Reported as a limitation and checked via (c).
- Models that fail to load from the cache are dropped and listed.

## Success Criteria
Every hypothesis component gets a number with a CI and a supported / not supported / mixed verdict; raw
per-prompt outputs are saved; the pipeline reruns from one command.

## Deviations from plan
- **Models.** Final set: Qwen2.5 1.5B/7B/14B, Qwen3-4B-Instruct-2507, Qwen3-8B, Llama-3.1-8B-Instruct (unsloth
  mirror, because the meta-llama snapshot in the cache had no config), Phi-3-mini. Gemma was dropped: tokenizer
  files missing locally and the repo is gated for the available token.
- **Label ablation excludes Phi-3**: it emits a whitespace token before digits, so first-token read-out is empty.
- **Added after seeing results** (not pre-registered, labelled as such): permutation null for the spread;
  decomposition of the as-published gain into key-placement and original-order parts; the regex split of MMLU
  into order-dependent vs self-contained options; split-half replication; low-label-mass filter.
- **Added metric**: "directed gap" (accuracy at a model's overall best minus overall worst position) in the
  difficulty strata, to avoid picking the best position separately inside each small stratum.
- **Not done**: prompt-paraphrase and few-shot ablations; hosted frontier models (no API key).
- Qwen3-4B's first full run died with an out-of-memory error caused by another job on the shared GPU and was rerun.
