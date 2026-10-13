# Resources actually used

The resource-finding stage was skipped for this workspace, so nothing was pre-gathered. Everything below was
located during the experiment session.

## Datasets
- `cais/mmlu` (config `all`, test split) — local HuggingFace cache.
- `tau/commonsense_qa` (validation split) — local HuggingFace cache.
- `google/boolq` (validation split) — downloaded from the HuggingFace Hub during the session.

Re-download: `datasets.load_dataset("cais/mmlu", "all")`, `load_dataset("tau/commonsense_qa")`,
`load_dataset("google/boolq")`. `datasets/items.parquet` holds the exact prompts built from them.

## Models (local HuggingFace cache, no downloads)
`Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`,
`Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-8B`, `unsloth/Meta-Llama-3.1-8B-Instruct`,
`microsoft/Phi-3-mini-4k-instruct`.

## Papers
None were downloaded. Works cited in REPORT.md are cited from memory.

## Code
No external repositories. All code is in `src/`.
