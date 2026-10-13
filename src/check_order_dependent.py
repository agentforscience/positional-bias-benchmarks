"""Robustness: is the 'published order is easier' effect on MMLU driven by options that only make
sense in their original place ("All of the above", "Both A and B", ...)?  Splits MMLU questions by a
regex flag and recomputes the as-published gain and the position spread on each part."""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from analyze import MODEL_ORDER, matrices, metrics, load

PAT = re.compile(r"\b(all|none|both|neither|either|any|each) of (the|these|those)\b|\babove\b|\bboth\b.*\band\b|"
                 r"\b[A-D] (and|or) [A-D]\b|\b(options?|choices?|answers?) [A-D]\b|\b\(?[a-d]\) (and|or)\b", re.I)

items = pd.read_parquet("datasets/items.parquet")
it = items[(items.cond == "mmlu_k4") & (items.rot == 0)]
flag = {q: any(PAT.search(o) for o in json.loads(opts)) for q, opts in zip(it.qid, it.options)}
print("order-dependent questions:", sum(flag.values()), "of", len(flag))

df = load()
rng = np.random.default_rng(42)
rows = []
for m in [m for m in MODEL_ORDER if m in set(df.model)]:
    q, C, P, G, og, k = matrices(df[(df.model == m) & (df.cond == "mmlu_k4")])
    f = np.array([flag[x] for x in q])
    for name, sel in (("order-dependent options", f), ("self-contained options", ~f)):
        mt = metrics(C[sel], P[sel], og[sel], k, True)
        rows.append(dict(model=m, subset=name, n_q=int(sel.sum()), bal_acc=mt["bal_acc"], spread=mt["spread"],
                         published_gain=mt["published_gain"], published_order_part=mt["published_order_part"]))
out = pd.DataFrame(rows)
out.to_csv("results/evaluations/order_dependent_check.csv", index=False)
print(out.round(2).to_string(index=False))

# pooled over models with a question bootstrap, self-contained subset
mats = {m: matrices(df[(df.model == m) & (df.cond == "mmlu_k4")]) for m in out.model.unique()}
for name, want in (("order-dependent options", True), ("self-contained options", False)):
    q = next(iter(mats.values()))[0]
    idx = np.where(np.array([flag[x] for x in q]) == want)[0]
    def f(i):
        ms = [metrics(C[i], P[i], og[i], 4, True) for _, C, P, _, og, _ in mats.values()]
        return [np.mean([x[key] for x in ms]) for key in ("published_gain", "spread", "bal_acc")]
    pt = f(idx)
    bs = np.array([f(rng.choice(idx, len(idx))) for _ in range(1000)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    print(f"{name}: n={len(idx)}  published_gain {pt[0]:.2f} [{lo[0]:.2f}, {hi[0]:.2f}]  "
          f"spread {pt[1]:.2f} [{lo[1]:.2f}, {hi[1]:.2f}]  bal_acc {pt[2]:.1f}")
