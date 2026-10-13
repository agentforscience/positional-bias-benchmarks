"""Run one local HuggingFace chat model over datasets/items.parquet.

For every prompt we take the next-token logits at the first assistant position and
keep the log-probabilities of the k label tokens (e.g. A-D). The prediction is the
argmax over labels. Nothing is sampled, so the output is deterministic.

Usage:
    python src/run_model.py --model Qwen/Qwen2.5-7B-Instruct --tag qwen2.5-7b [--limit 200]
    python src/run_model.py --model ... --tag ... --generate 300   # E6: greedy text read-out check
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42
OUT_DIR = "results/model_outputs"


_VOCAB_STR = {}


def label_token_ids(tok, labels):
    """All vocabulary ids whose decoded text is exactly the label, ignoring leading whitespace.

    Covers "A", " A" and SentencePiece "\u2581A" without assuming how the tokenizer splits them.
    """
    if id(tok) not in _VOCAB_STR:
        _VOCAB_STR[id(tok)] = [tok.decode([i]).lstrip() for i in range(len(tok))]
    vocab = _VOCAB_STR[id(tok)]
    ids = []
    for lab in labels:
        cands = [i for i, s in enumerate(vocab) if s == lab]
        assert cands, f"label {lab!r} has no single-token form"
        ids.append(cands)
    return ids


def build_chat(tok, prompt):
    msgs = [{"role": "user", "content": prompt}]
    try:  # Qwen3 templates open a <think> block unless told otherwise
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--items", default="datasets/items.parquet")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N prompts per condition")
    ap.add_argument("--generate", type=int, default=0, help="greedy-generate N mmlu_k4 prompts instead")
    ap.add_argument("--max_tokens_per_batch", type=int, default=12000)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(args.items)
    if args.limit:
        df = df.groupby("cond").head(args.limit)
    if args.generate:
        df = df[df.cond == "mmlu_k4"].sample(n=args.generate, random_state=SEED)

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16, device_map={"": 0}
    ).eval()

    texts = [build_chat(tok, p) for p in df.prompt]
    # the template already contains BOS where the model needs one
    enc_len = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
    order = np.argsort(enc_len)[::-1]
    print(f"{args.tag}: {len(df)} prompts, max len {max(enc_len)} tokens", flush=True)
    print("---- rendered example ----\n" + texts[0][-400:] + "\n--------------------------", flush=True)

    labels_all = [json.loads(s) for s in df.labels]
    id_cache = {}
    records = [None] * len(df)
    t0 = time.time()
    i = 0
    while i < len(order):
        bs = max(1, min(96, args.max_tokens_per_batch // enc_len[order[i]]))
        idx = order[i : i + bs]
        i += bs
        batch = tok([texts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(0)
        if args.generate:
            out = model.generate(**batch, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
            gen = tok.batch_decode(out[:, batch.input_ids.shape[1]:], skip_special_tokens=True)
            for j, g in zip(idx, gen):
                records[j] = dict(generated=g)
            continue
        logits = model(**batch).logits[:, -1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        for row, j in enumerate(idx):
            labs = labels_all[j]
            key = tuple(labs)
            if key not in id_cache:
                id_cache[key] = label_token_ids(tok, labs)
            # a label's score is the log of the total probability over its token variants
            lp = [torch.logsumexp(logp[row, ids], dim=0).item() for ids in id_cache[key]]
            records[j] = dict(
                label_logprobs=json.dumps([round(x, 5) for x in lp]),
                label_mass=float(np.exp(lp).sum()),
                pred_pos=int(np.argmax(lp)),
                top_token=tok.decode(int(logits[row].argmax())),
            )
        if (i // bs) % 50 == 0:
            print(f"  {i}/{len(order)}  {time.time() - t0:.0f}s", flush=True)

    res = pd.concat([df.drop(columns=["prompt", "options"]).reset_index(drop=True), pd.DataFrame(records)], axis=1)
    res["model"] = args.tag
    if args.generate:
        res["gen_letter"] = res.generated.map(lambda s: (re.findall(r"\b([A-D])\b", s) or [None])[0])
        path = f"{OUT_DIR}/{args.tag}__generate.parquet"
    else:
        res["correct"] = (res.pred_pos == res.gold_pos).astype(int)
        path = f"{OUT_DIR}/{args.tag}{'__smoke' if args.limit else ''}.parquet"
        print(res.groupby("cond").agg(acc=("correct", "mean"), label_mass=("label_mass", "mean")).round(3))
    res.to_parquet(path)
    meta = dict(model=args.model, tag=args.tag, n=len(res), seconds=round(time.time() - t0, 1),
                dtype="bfloat16", torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                revision=getattr(model.config, "_commit_hash", None))
    json.dump(meta, open(path.replace(".parquet", ".meta.json"), "w"), indent=2)
    print("saved", path, meta, flush=True)


if __name__ == "__main__":
    main()
