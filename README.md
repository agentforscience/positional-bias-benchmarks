# Positional Bias in Multiple-Choice Benchmarks

Tests whether LLMs favour early answer positions and how much that changes multiple-choice scores. Seven
open-weight instruct models (1.5B-14B) answer MMLU, CommonsenseQA and BoolQ questions with the correct option
rotated through every position, at 2-5 options and with letter, number and True/False labels.

Full write-up: [REPORT.md](REPORT.md). Plan written before the runs: [planning.md](planning.md).

## Key findings

- Position matters but there is no shared preference for A/B. On 4-option MMLU the gap between a model's best
  and worst key position is 4-8 points for six of seven models; five of seven do worse than average at A.
- Published scores are barely inflated: +1.3 points on MMLU [0.7, 2.0] and +0.3 on CommonsenseQA [-0.3, 1.0]
  over position-balanced accuracy. On MMLU none of this comes from key placement.
- True/false is not the safest format. Answering with the word True/False gave a mean 14.2-point gap between
  the two orders (51 points for the 1.5B model). Five options averaged 8.8; three of seven models exceeded 10.
- The effect is near zero on easy questions and peaks where a model is around 50% accuracy. About 30% of MMLU
  questions change from right to wrong under some option order.
- Relabelling A-D as 1-4 more than doubled the gap (to 13.6 points), so the effect is tied to label symbols.

## Caveats to read first

- No API key was available, so no hosted frontier model was run. Results cover local models of 14B or fewer.
- The pre-gathered literature review, datasets and code were absent; background is from memory.

## Reproduce

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv sync --no-install-project            # or: uv pip install -r requirements.txt
export HF_HOME=/path/to/hf/cache        # models are loaded with local_files_only=True

python src/build_items.py               # writes datasets/items.parquet (42,594 prompts)
CUDA_VISIBLE_DEVICES=0 python src/run_model.py --model Qwen/Qwen2.5-7B-Instruct --tag qwen2.5-7b
CUDA_VISIBLE_DEVICES=0 python src/run_model.py --model Qwen/Qwen2.5-7B-Instruct --tag qwen2.5-7b --generate 400
python src/analyze.py > results/evaluations/analysis_output.txt
PYTHONPATH=src python src/check_order_dependent.py
```

Model repos and tags are listed in REPORT.md section 3. One model takes 4-30 minutes on an RTX A6000; the
analysis takes about 10 minutes on CPU. Adding a model means adding its tag to `MODEL_ORDER` in `src/analyze.py`.

## Files

| Path | Contents |
|---|---|
| `src/build_items.py` | Builds every prompt (rotations, reduced option counts, label variants) |
| `src/run_model.py` | Runs one model; saves label log-probabilities per prompt |
| `src/analyze.py` | Metrics, bootstrap intervals, tests, figures |
| `src/check_order_dependent.py` | Splits MMLU by "All of the above"-style options |
| `datasets/items.parquet`, `datasets/example_items.jsonl` | Prompts used; a few examples per condition |
| `results/model_outputs/` | Raw per-prompt outputs and run metadata for each model |
| `results/evaluations/` | Result tables (CSV) and the full analysis log |
| `figures/` | Five figures used in the report |
| `logs/runs/` | Run logs |
