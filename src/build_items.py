"""Build every prompt used in the study and write them to datasets/items.parquet.

Each question is shown once per cyclic rotation of its option list, so the gold
answer sits at every position exactly once. Reduced-k conditions keep the gold
option plus a seeded, nested sample of distractors from the same question.

Conditions
    mmlu_k4 / mmlu_k3 / mmlu_k2   MMLU test sample, letter labels
    mmlu_k4_num                   same as mmlu_k4 with labels 1-4 (label ablation)
    csqa_k5 / k4 / k3 / k2        CommonsenseQA validation, letter labels
    boolq_ab                      BoolQ as "A. True / B. False" (both orders)
    boolq_word                    BoolQ answered with the word True/False (both orders)
"""
import glob
import json
import os

import numpy as np
import pandas as pd

SEED = 42
N_MMLU = 1500
N_BOOLQ = 1500
LETTERS = list("ABCDE")
OUT = "datasets/items.parquet"

MC_TEMPLATE = "{question}\n\n{options}\n\nAnswer with the {kind} of the correct option only."
BOOL_AB_TEMPLATE = (
    "Passage: {passage}\n\nStatement: The answer to the question \"{question}?\" is yes.\n\n"
    "{options}\n\nAnswer with the letter of the correct option only."
)
BOOL_WORD_TEMPLATE = (
    "Passage: {passage}\n\nStatement: The answer to the question \"{question}?\" is yes.\n\n"
    "Is the statement {first} or {second}? Answer with one word only: {first} or {second}."
)


def hub(pattern):
    """Resolve a file inside the local HuggingFace cache."""
    hits = glob.glob(os.path.join(os.environ["HF_HOME"], "hub", pattern))
    assert hits, f"not in HF cache: {pattern}"
    return hits[0]


def rotations(options, gold_idx, labels, cond, qid, meta, question, kind):
    """Yield one prompt row per cyclic rotation of `options`."""
    k = len(options)
    for rot in range(k):
        order = [(i + rot) % k for i in range(k)]
        opts = [options[i] for i in order]
        gold_pos = order.index(gold_idx)
        lines = "\n".join(f"{labels[j]}. {opts[j]}" for j in range(k))
        yield dict(
            cond=cond, qid=qid, rot=rot, k=k, gold_pos=gold_pos,
            labels=json.dumps(labels[:k]), options=json.dumps(opts),
            prompt=MC_TEMPLATE.format(question=question.strip(), options=lines, kind=kind),
            **meta,
        )


def reduce_options(options, gold_idx, k, rng_order):
    """Keep gold + first (k-1) distractors of a fixed random distractor order; preserve original order."""
    keep = sorted([gold_idx] + list(rng_order[: k - 1]))
    return [options[i] for i in keep], keep.index(gold_idx)


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    # ---- MMLU: stratified sample over the 57 subjects --------------------------------
    mmlu = pd.read_parquet(hub("datasets--cais--mmlu/snapshots/*/all/test-*.parquet"))
    mmlu = mmlu.drop_duplicates(subset=["question"]).reset_index(drop=True)
    mmlu["src_idx"] = mmlu.index
    frac = N_MMLU / len(mmlu)
    samp = mmlu.groupby("subject", group_keys=False).sample(frac=frac, random_state=SEED)
    samp = samp.sample(n=min(N_MMLU, len(samp)), random_state=SEED)
    for r in samp.itertuples():
        opts = [str(c) for c in r.choices]
        assert len(opts) == 4 and len(set(opts)) == 4 or len(opts) == 4
        qid = f"mmlu_{r.src_idx}"
        meta = dict(source="mmlu", subject=r.subject, orig_gold=int(r.answer))
        distract = [i for i in range(4) if i != r.answer]
        dorder = rng.permutation(distract)
        for k in (4, 3, 2):
            o, g = reduce_options(opts, int(r.answer), k, dorder)
            rows += rotations(o, g, LETTERS, f"mmlu_k{k}", qid, meta, r.question, "letter")
        rows += rotations(opts, int(r.answer), list("1234"), "mmlu_k4_num", qid, meta, r.question, "number")

    # ---- CommonsenseQA validation (all) ------------------------------------------------
    csqa = pd.read_parquet(hub("datasets--tau--commonsense_qa/snapshots/*/data/validation-*.parquet"))
    for r in csqa.itertuples():
        opts = [str(t) for t in r.choices["text"]]
        gold = list(r.choices["label"]).index(r.answerKey)
        qid = f"csqa_{r.id}"
        meta = dict(source="csqa", subject=r.question_concept, orig_gold=gold)
        dorder = rng.permutation([i for i in range(5) if i != gold])
        for k in (5, 4, 3, 2):
            o, g = reduce_options(opts, gold, k, dorder)
            rows += rotations(o, g, LETTERS, f"csqa_k{k}", qid, meta, r.question, "letter")

    # ---- BoolQ validation sample: native true/false ------------------------------------
    from datasets import load_dataset

    boolq = load_dataset("google/boolq", split="validation").to_pandas()
    boolq["src_idx"] = boolq.index
    boolq = boolq[boolq.passage.str.len() < 1500].sample(n=N_BOOLQ, random_state=SEED)
    for r in boolq.itertuples():
        qid = f"boolq_{r.src_idx}"
        truth = "True" if r.answer else "False"
        meta = dict(source="boolq", subject="boolq", orig_gold=0 if r.answer else 1)
        q = r.question.strip().rstrip("?")
        for rot, order in enumerate((["True", "False"], ["False", "True"])):
            gold_pos = order.index(truth)
            lines = "\n".join(f"{LETTERS[j]}. {order[j]}" for j in range(2))
            rows.append(dict(
                cond="boolq_ab", qid=qid, rot=rot, k=2, gold_pos=gold_pos,
                labels=json.dumps(LETTERS[:2]), options=json.dumps(order),
                prompt=BOOL_AB_TEMPLATE.format(passage=r.passage, question=q, options=lines), **meta))
            rows.append(dict(
                cond="boolq_word", qid=qid, rot=rot, k=2, gold_pos=gold_pos,
                labels=json.dumps(order), options=json.dumps(order),
                prompt=BOOL_WORD_TEMPLATE.format(passage=r.passage, question=q, first=order[0], second=order[1]),
                **meta))

    df = pd.DataFrame(rows)
    df["item_id"] = np.arange(len(df))
    # every question must have gold at every position exactly once
    chk = df.groupby(["cond", "qid"]).gold_pos.agg(lambda s: sorted(s) == list(range(len(s))))
    assert chk.all(), "rotation design broken"
    os.makedirs("datasets", exist_ok=True)
    df.to_parquet(OUT)
    print(df.groupby("cond").agg(prompts=("qid", "size"), questions=("qid", "nunique"), k=("k", "first")))
    print("total prompts", len(df))
    for c in ("mmlu_k4", "csqa_k5", "boolq_ab", "boolq_word", "mmlu_k4_num"):
        print(f"\n===== example {c} =====\n" + df[df.cond == c].iloc[1].prompt[-600:])
    df.groupby("cond").head(4).to_json("datasets/example_items.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
