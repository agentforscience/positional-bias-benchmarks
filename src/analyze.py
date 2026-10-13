"""Analysis for the positional-bias study. Reads results/model_outputs/*.parquet,
writes tables to results/evaluations/ and figures to figures/.

All confidence intervals are question-level (cluster) bootstraps: a question's k
rotations are resampled together. Accuracies and gaps are in percentage points.
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.multitest import multipletests

SEED = 42
N_BOOT = 2000
EV = "results/evaluations"
FIG = "figures"
MODEL_ORDER = ["qwen2.5-1.5b", "qwen2.5-7b", "qwen2.5-14b", "qwen3-4b-inst", "qwen3-8b", "llama3.1-8b", "phi3-mini"]
COND_ORDER = ["boolq_word", "boolq_ab", "mmlu_k2", "mmlu_k3", "mmlu_k4", "mmlu_k4_num",
              "csqa_k2", "csqa_k3", "csqa_k4", "csqa_k5"]
NATIVE = {"mmlu_k4", "mmlu_k4_num", "csqa_k5"}  # rotation 0 == as-published option order
MIN_LABEL_MASS = 0.5  # a model x condition whose mean label mass is below this is not readable by first token

# dataviz palette: fixed categorical order for models, one-hue ramp for ordered positions
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
MCOL = dict(zip(MODEL_ORDER, SERIES))
POS_RAMP = ["#0d366b", "#256abf", "#5598e7", "#86b6ef", "#b7d3f6"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})


# ----------------------------------------------------------------------------- loading
def load():
    files = [f for f in glob.glob("results/model_outputs/*.parquet") if "__" not in os.path.basename(f)]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["model"] = pd.Categorical(df.model, [m for m in MODEL_ORDER if m in set(df.model)])
    lp = df.label_logprobs.map(json.loads)
    # probability of the gold option renormalised over the k labels
    df["p_gold"] = [np.exp(l[g]) / np.exp(l).sum() for l, g in zip(lp, df.gold_pos)]
    return df


def matrices(d):
    """Question x gold-position matrices for one model x condition."""
    d = d.sort_values(["qid", "gold_pos"])
    k = int(d.k.iloc[0])
    q = d.qid.to_numpy(dtype=object).reshape(-1, k)[:, 0]
    C = d.correct.values.reshape(-1, k).astype(float)       # correct when gold at position p
    P = d.pred_pos.values.reshape(-1, k)                    # chosen position
    G = d.p_gold.values.reshape(-1, k)
    og = d.orig_gold.values.reshape(-1, k)[:, 0]
    return q, C, P, G, og, k


def metrics(C, P, og, k, native, best=None, worst=None):
    """Point estimates from a question x position correctness matrix."""
    acc_p = C.mean(0) * 100
    bal = acc_p.mean()
    out = {"bal_acc": bal, "spread": acc_p.max() - acc_p.min(), "rstd": acc_p.std(),
           "first_gain": acc_p[0] - bal, "last_gain": acc_p[-1] - bal, "best_gain": acc_p.max() - bal,
           "flip_rate": ((C.max(1) != C.min(1)).mean()) * 100}
    if k >= 4:
        out["early2_gain"] = acc_p[:2].mean() - bal
    if native:
        out["published_gain"] = C[np.arange(len(C)), og].mean() * 100 - bal
        # part explained by where the benchmark puts its keys: sum_p w_p * acc_p - bal
        w = np.bincount(og, minlength=k) / len(og)
        out["published_key_part"] = float(w @ acc_p) - bal
        # remainder: the published arrangement itself is easier than a rotation with the same key position
        out["published_order_part"] = out["published_gain"] - out["published_key_part"]
    if best is not None:  # gap between positions fixed in advance (no max-selection bias)
        out["directed_gap"] = acc_p[best] - acc_p[worst]
    for p in range(k):
        out[f"acc_{p}"] = acc_p[p]
        out[f"share_{p}"] = (P == p).mean() * 100
    return out


def boot(fn, n, rng, n_boot=N_BOOT):
    """Percentile bootstrap over question indices. fn(idx) -> dict of floats."""
    point = fn(np.arange(n))
    draws = [fn(rng.integers(0, n, n)) for _ in range(n_boot)]
    lo = {m: np.percentile([d[m] for d in draws], 2.5) for m in point}
    hi = {m: np.percentile([d[m] for d in draws], 97.5) for m in point}
    return point, lo, hi


def fmt(p, lo, hi, key, nd=1):
    return f"{p[key]:.{nd}f} [{lo[key]:.{nd}f}, {hi[key]:.{nd}f}]"


# ----------------------------------------------------------------------------- E1/E2/E3/E5
def main_table(df, rng):
    rows, tests = [], []
    for (model, cond), d in df.groupby(["model", "cond"], observed=True):
        q, C, P, G, og, k = matrices(d)
        native = cond in NATIVE
        mass = d.label_mass.mean()
        p, lo, hi = boot(lambda i: metrics(C[i], P[i], og[i], k, native), len(C), rng)
        r = dict(model=model, cond=cond, k=k, n_q=len(C), label_mass=mass, valid=mass >= MIN_LABEL_MASS)
        for m in p:
            r[m], r[m + "_lo"], r[m + "_hi"] = p[m], lo[m], hi[m]
        # Null reference: shuffle position labels within each question (keeps flip structure, removes position).
        null = np.array([[m_["spread"], m_["best_gain"]] for m_ in (
            metrics(rng.permuted(C, axis=1), P, og, k, False) for _ in range(300))])
        r.update(null_spread=null[:, 0].mean(), null_best_gain=null[:, 1].mean(),
                 perm_p_spread=(1 + (null[:, 0] >= p["spread"]).sum()) / 301)
        # Cochran's Q: does correctness depend on gold position?
        cq = cochrans_q(C.astype(int))
        # chi-square on chosen-position counts vs uniform (key is balanced by design)
        counts = np.bincount(P.ravel(), minlength=k)
        chi = stats.chisquare(counts)
        r.update(cochran_q=cq.statistic, cochran_p=cq.pvalue, chi2=chi.statistic, chi2_p=chi.pvalue,
                 top_pos="ABCDE"[int(np.argmax(counts))], best_pos="ABCDE"[int(np.argmax(C.mean(0)))],
                 worst_pos="ABCDE"[int(np.argmin(C.mean(0)))])
        rows.append(r)
    t = pd.DataFrame(rows)
    for col in ("cochran_p", "chi2_p"):
        t[col + "_holm"] = multipletests(t[col].fillna(1.0), method="holm")[1]
    t["cond"] = pd.Categorical(t.cond, COND_ORDER)
    t = t.sort_values(["cond", "model"]).reset_index(drop=True)
    t.to_csv(f"{EV}/main_table.csv", index=False)
    return t


def pooled_by_cond(df, t, rng):
    """Mean over models of each metric, with a bootstrap over the shared questions."""
    rows = []
    for cond in COND_ORDER:
        ok = t[(t.cond == cond) & t.valid].model.tolist()
        mats = [matrices(df[(df.model == m) & (df.cond == cond)]) for m in ok]
        k, n = mats[0][5], len(mats[0][1])
        assert all((mm[0] == mats[0][0]).all() for mm in mats)

        def f(i):
            ms = [metrics(C[i], P[i], og[i], k, cond in NATIVE) for _, C, P, _, og, _ in mats]
            return {key: float(np.mean([m[key] for m in ms])) for key in ms[0]}

        p, lo, hi = boot(f, n, rng, 1000)
        r = dict(cond=cond, k=k, n_models=len(ok), n_q=n)
        for m in p:
            r[m], r[m + "_lo"], r[m + "_hi"] = p[m], lo[m], hi[m]
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(f"{EV}/pooled_by_condition.csv", index=False)
    return out


def format_contrasts(df, t, rng):
    """Paired (same questions, same model) differences in spread between option counts."""
    pairs = [("mmlu_k4", "mmlu_k2"), ("mmlu_k4", "mmlu_k3"), ("csqa_k5", "csqa_k2"), ("csqa_k5", "csqa_k4"),
             ("csqa_k4", "csqa_k2"), ("mmlu_k4", "mmlu_k4_num")]
    rows = []
    for a, b in pairs:
        ok = sorted(set(t[(t.cond == a) & t.valid].model) & set(t[(t.cond == b) & t.valid].model))
        ma = [matrices(df[(df.model == m) & (df.cond == a)]) for m in ok]
        mb = [matrices(df[(df.model == m) & (df.cond == b)]) for m in ok]
        n = len(ma[0][1])

        def f(i):
            out = {}
            for key in ("spread", "flip_rate", "best_gain"):
                va = np.mean([metrics(x[1][i], x[2][i], x[4][i], x[5], False)[key] for x in ma])
                vb = np.mean([metrics(x[1][i], x[2][i], x[4][i], x[5], False)[key] for x in mb])
                out[key] = va - vb
            return out

        p, lo, hi = boot(f, n, rng, 1000)
        rows.append(dict(contrast=f"{a} - {b}", n_models=len(ok),
                         **{m: fmt(p, lo, hi, m) for m in p}))
    out = pd.DataFrame(rows)
    out.to_csv(f"{EV}/format_contrasts.csv", index=False)
    return out


# ----------------------------------------------------------------------------- E4 difficulty
def difficulty_tables(df, t, rng):
    """Stratify by difficulty that is NOT computed from the responses being analysed.

    loo  : mean rotation-balanced correctness of the other models on the question
    own  : the model's own rotation-averaged p(gold) under numeric labels (MMLU only)
    """
    strata_rows, pooled_rows = [], []
    for cond in ("mmlu_k4", "csqa_k5"):
        models = t[(t.cond == cond) & t.valid].model.tolist()
        M = {m: matrices(df[(df.model == m) & (df.cond == cond)]) for m in models}
        q0 = M[models[0]][0]
        k = M[models[0]][5]
        qbal = np.stack([M[m][1].mean(1) for m in models])            # model x question
        schemes = {"loo": {m: (qbal.sum(0) - qbal[j]) / (len(models) - 1) for j, m in enumerate(models)}}
        if cond == "mmlu_k4":
            own = {}
            for m in models:
                row = t[(t.cond == "mmlu_k4_num") & (t.model == m)]
                if len(row) and bool(row.valid.iloc[0]):
                    qn, _, _, Gn, _, _ = matrices(df[(df.model == m) & (df.cond == "mmlu_k4_num")])
                    assert (qn == q0).all()
                    own[m] = Gn.mean(1)
            schemes["own_numlabel"] = own
        for scheme, diff in schemes.items():
            for nbin, names in ((3, ["hard", "medium", "easy"]), (5, ["Q1 hardest", "Q2", "Q3", "Q4", "Q5 easiest"])):
                bins = {}
                for m, dvals in diff.items():
                    # rank-based equal-count bins; ties broken by a fixed random jitter
                    jit = np.random.default_rng(SEED).random(len(dvals)) * 1e-6
                    bins[m] = pd.qcut(dvals + jit, nbin, labels=False)
                for b in range(nbin):
                    per_model = {}
                    for m in bins:
                        _, C, P, _, og, _ = M[m]
                        best, worst = int(np.argmax(C.mean(0))), int(np.argmin(C.mean(0)))
                        sel = np.where(bins[m] == b)[0]
                        per_model[m] = (C[sel], P[sel], og[sel], best, worst)
                        mt = metrics(C[sel], P[sel], og[sel], k, True, best, worst)
                        mt["attrib_share_best"] = 100 * mt["best_gain"] / max(C[sel].mean(0).max() * 100, 1e-9)
                        strata_rows.append(dict(cond=cond, scheme=scheme, nbin=nbin, bin=b, bin_name=names[b],
                                                model=m, n_q=len(sel), **mt))

                    def f(_):
                        ms = []
                        for m, (C, P, og, best, worst) in per_model.items():
                            i = rng.integers(0, len(C), len(C)) if _ is not None else np.arange(len(C))
                            mt = metrics(C[i], P[i], og[i], k, True, best, worst)
                            top = C[i].mean(0).max() * 100
                            mt["attrib_share_best"] = 100 * mt["best_gain"] / max(top, 1e-9)
                            mt["attrib_share_published"] = 100 * mt["published_gain"] / max(mt["bal_acc"] + mt["published_gain"], 1e-9)
                            ms.append(mt)
                        return {key: float(np.mean([x[key] for x in ms])) for key in ms[0]}

                    point = f(None)
                    draws = [f(1) for _ in range(1000)]
                    r = dict(cond=cond, scheme=scheme, nbin=nbin, bin=b, bin_name=names[b], n_models=len(bins))
                    for key in ("bal_acc", "spread", "directed_gap", "flip_rate", "best_gain", "first_gain",
                                "published_gain", "attrib_share_best", "attrib_share_published"):
                        vals = [d[key] for d in draws]
                        r[key], r[key + "_lo"], r[key + "_hi"] = point[key], np.percentile(vals, 2.5), np.percentile(vals, 97.5)
                        r["_draws_" + key] = vals
                    pooled_rows.append(r)
    strata = pd.DataFrame(strata_rows)
    pooled = pd.DataFrame(pooled_rows)
    # medium vs extremes (terciles): strata are disjoint question sets, so draws are independent
    cmp_rows = []
    for (cond, scheme), g in pooled[pooled.nbin == 3].groupby(["cond", "scheme"]):
        g = g.set_index("bin_name")
        for key in ("spread", "directed_gap", "flip_rate"):
            med = np.array(g.loc["medium", "_draws_" + key])
            for other in ("easy", "hard"):
                dlt = med - np.array(g.loc[other, "_draws_" + key])
                cmp_rows.append(dict(cond=cond, scheme=scheme, metric=key, contrast=f"medium - {other}",
                                     diff=g.loc["medium", key] - g.loc[other, key],
                                     lo=np.percentile(dlt, 2.5), hi=np.percentile(dlt, 97.5)))
    cmp = pd.DataFrame(cmp_rows)
    pooled = pooled[[c for c in pooled.columns if not c.startswith("_draws_")]]
    strata.to_csv(f"{EV}/difficulty_by_model.csv", index=False)
    pooled.to_csv(f"{EV}/difficulty_pooled.csv", index=False)
    cmp.to_csv(f"{EV}/difficulty_contrasts.csv", index=False)
    return strata, pooled, cmp


# ----------------------------------------------------------------------------- robustness
def readout_check(df):
    rows = []
    for f in sorted(glob.glob("results/model_outputs/*__generate.parquet")):
        g = pd.read_parquet(f)
        m = g.model.iloc[0]
        if not ((df.model == m) & (df.cond == "mmlu_k4")).any():
            continue
        base = df[(df.model == m) & (df.cond == "mmlu_k4")].set_index("item_id")
        g = g.set_index("item_id")
        logit_letter = base.loc[g.index, "pred_pos"].map(lambda p: "ABCD"[p])
        parsed = g.gen_letter.notna()
        rows.append(dict(model=m, n=len(g), parsed_pct=100 * parsed.mean(),
                         agree_pct=100 * (g.gen_letter[parsed] == logit_letter[parsed]).mean(),
                         agree_all_pct=100 * (g.gen_letter.fillna("?") == logit_letter).mean()))
    out = pd.DataFrame(rows)
    out.to_csv(f"{EV}/readout_check.csv", index=False)
    return out


def low_mass_robustness(df):
    """Recompute headline 4-option numbers after dropping questions with any low-label-mass rotation."""
    rows = []
    for (model, cond), d in df[df.cond.isin(["mmlu_k4", "csqa_k5"])].groupby(["model", "cond"], observed=True):
        bad = d.groupby("qid").label_mass.min() < 0.5
        keep = d[~d.qid.isin(bad[bad].index)]
        for name, dd in (("all", d), ("mass>=0.5", keep)):
            q, C, P, G, og, k = matrices(dd)
            m = metrics(C, P, og, k, True)
            rows.append(dict(model=model, cond=cond, subset=name, n_q=len(C), bal_acc=m["bal_acc"],
                             spread=m["spread"], published_gain=m["published_gain"], first_gain=m["first_gain"]))
    out = pd.DataFrame(rows)
    out.to_csv(f"{EV}/low_mass_robustness.csv", index=False)
    return out


def subject_split(df):
    """Does the direction of the bias replicate across two disjoint halves of the questions?"""
    rows = []
    rng = np.random.default_rng(SEED)
    for (model, cond), d in df[df.cond.isin(["mmlu_k4", "csqa_k5"])].groupby(["model", "cond"], observed=True):
        q, C, P, G, og, k = matrices(d)
        perm = rng.permutation(len(C))
        a, b = perm[: len(C) // 2], perm[len(C) // 2:]
        rows.append(dict(model=model, cond=cond,
                         r_acc_by_pos=np.corrcoef(C[a].mean(0), C[b].mean(0))[0, 1],
                         best_a="ABCDE"[C[a].mean(0).argmax()], best_b="ABCDE"[C[b].mean(0).argmax()],
                         worst_a="ABCDE"[C[a].mean(0).argmin()], worst_b="ABCDE"[C[b].mean(0).argmin()]))
    out = pd.DataFrame(rows)
    out.to_csv(f"{EV}/split_half.csv", index=False)
    return out


# ----------------------------------------------------------------------------- figures
def fig_acc_by_position(t):
    conds = [("mmlu_k4", "MMLU, 4 options"), ("csqa_k5", "CommonsenseQA, 5 options")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for ax, (cond, title) in zip(axes, conds):
        g = t[(t.cond == cond) & t.valid]
        k = int(g.k.iloc[0])
        for _, r in g.iterrows():
            y = [r[f"acc_{p}"] - r.bal_acc for p in range(k)]
            lo = [r[f"acc_{p}"] - r[f"acc_{p}_lo"] for p in range(k)]
            hi = [r[f"acc_{p}_hi"] - r[f"acc_{p}"] for p in range(k)]
            ax.errorbar(range(k), y, yerr=[lo, hi], color=MCOL[r.model], lw=2, marker="o", ms=6,
                        mec=SURFACE, mew=1.2, capsize=2, elinewidth=0.8, label=r.model)
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_xticks(range(k), list("ABCDE"[:k]))
        ax.set_xlabel("Position of the correct answer")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Accuracy minus the model's\nposition-balanced accuracy (points)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="Model")
    fig.suptitle("Accuracy depends on where the correct answer sits, but not in one shared direction",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig1_accuracy_by_gold_position.png", dpi=170)
    plt.close(fig)


def fig_selection_share(t):
    """Deviation of each position's selection share from uniform, so bars are anchored at zero."""
    conds = [("mmlu_k4", "MMLU, 4 options (unbiased = 25% each)"), ("csqa_k5", "CommonsenseQA, 5 options (unbiased = 20% each)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (cond, title) in zip(axes, conds):
        g = t[(t.cond == cond) & t.valid].reset_index(drop=True)
        k = int(g.k.iloc[0])
        w = 0.8 / k
        for p in range(k):
            x = np.arange(len(g)) + (p - (k - 1) / 2) * w
            y = g[f"share_{p}"] - 100 / k
            ax.bar(x, y, width=w * 0.86, color=POS_RAMP[p], label="ABCDE"[p])
            ax.errorbar(x, y, yerr=[g[f"share_{p}"] - g[f"share_{p}_lo"], g[f"share_{p}_hi"] - g[f"share_{p}"]],
                        fmt="none", ecolor="#52514e", elinewidth=0.7)
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_xticks(range(len(g)), g.model, rotation=25, ha="right")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of answers given to the position\nminus the unbiased share (points)")
    axes[1].legend(title="Position", loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("Over- and under-selection of each position when the answer key is perfectly balanced",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig2_selection_share.png", dpi=170)
    plt.close(fig)


def fig_format(t):
    """Spread and as-published gain across formats; one panel per metric (never two y-scales)."""
    order = ["boolq_word", "boolq_ab", "mmlu_k2", "mmlu_k3", "mmlu_k4", "csqa_k2", "csqa_k3", "csqa_k4", "csqa_k5"]
    labels = ["BoolQ\nTrue/False\nwords", "BoolQ\nA/B", "MMLU\nk=2", "MMLU\nk=3", "MMLU\nk=4",
              "CSQA\nk=2", "CSQA\nk=3", "CSQA\nk=4", "CSQA\nk=5"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, yl in zip(axes, ("spread", "flip_rate"),
                           ("Best minus worst gold position (accuracy points)",
                            "Questions whose correctness changes with order (%)")):
        for m in [m for m in MODEL_ORDER if m in set(t.model)]:
            g = t[(t.model == m) & t.valid].set_index("cond")
            xs = [i for i, c in enumerate(order) if c in g.index]
            ax.plot(xs, [g.loc[order[i], key] for i in xs], "o", color=MCOL[m], ms=7, mec=SURFACE, mew=1.2, label=m)
        means = [t[(t.cond == c) & t.valid][key].mean() for c in order]
        ax.plot(range(len(order)), means, "_", color=INK, ms=22, mew=2, label="mean of models")
        for x in (1.5, 4.5):
            ax.axvline(x, color=GRID, lw=1)
        ax.set_xticks(range(len(order)), labels)
        ax.set_ylabel(yl)
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
    axes[0].set_title("Position spread by format", loc="left")
    axes[1].set_title("Order-sensitive questions by format", loc="left")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig3_format_ladder.png", dpi=170)
    plt.close(fig)


def fig_difficulty(pooled):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    specs = [("directed_gap", "Gap between each model's best and worst\nposition (accuracy points)"),
             ("flip_rate", "Questions whose correctness\nchanges with order (%)"),
             ("attrib_share_best", "Share of best-position accuracy\nthat is positional (%)")]
    series = [("mmlu_k4", "loo", "MMLU k=4, difficulty from other models", SERIES[0], "o"),
              ("csqa_k5", "loo", "CSQA k=5, difficulty from other models", SERIES[1], "s"),
              ("mmlu_k4", "own_numlabel", "MMLU k=4, model's own confidence (numeric-label run)", SERIES[2], "^")]
    for ax, (key, yl) in zip(axes, specs):
        for cond, scheme, lab, col, mk in series:
            g = pooled[(pooled.cond == cond) & (pooled.scheme == scheme) & (pooled.nbin == 5)].sort_values("bin")
            ax.errorbar(range(5), g[key], yerr=[g[key] - g[key + "_lo"], g[key + "_hi"] - g[key]], color=col,
                        marker=mk, ms=7, lw=2, mec=SURFACE, mew=1.2, capsize=2, elinewidth=0.8, label=lab)
        ax.set_xticks(range(5), ["Q1\nhardest", "Q2", "Q3", "Q4", "Q5\neasiest"])
        ax.set_ylabel(yl)
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    axes[0].set_title("Position effect by question difficulty (mean of models, 95% CI)", loc="left")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig4_difficulty.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_inflation(t):
    g = t[t.cond.isin(["mmlu_k4", "csqa_k5"]) & t.valid]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    defs = [("published_gain", "as-published order"), ("early2_gain", "key at A or B"), ("best_gain", "key at model's best position")]
    cols = [SERIES[0], SERIES[1], SERIES[2]]
    for ax, cond, title in zip(axes, ("mmlu_k4", "csqa_k5"), ("MMLU, 4 options", "CommonsenseQA, 5 options")):
        gg = g[g.cond == cond].reset_index(drop=True)
        for j, ((key, lab), col) in enumerate(zip(defs, cols)):
            x = np.arange(len(gg)) + (j - 1) * 0.26
            ax.bar(x, gg[key], width=0.22, color=col, label=lab)
            ax.errorbar(x, gg[key], yerr=[gg[key] - gg[key + "_lo"], gg[key + "_hi"] - gg[key]], fmt="none",
                        ecolor="#52514e", elinewidth=0.7)
        ax.axhspan(3, 8, color="#f0efec", zorder=0)
        ax.text(len(gg) - 0.45, 7.9, "hypothesised 3-8", ha="right", va="top", color=MUTED, fontsize=8)
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_xticks(range(len(gg)), gg.model, rotation=25, ha="right")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Accuracy minus position-\nbalanced accuracy (points)")
    axes[1].legend(title="Answer key placement", loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("Three definitions of score inflation from answer position", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig5_inflation_definitions.png", dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------------- driver
def main():
    os.makedirs(EV, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pd.set_option("display.width", 250, "display.max_columns", 60, "display.float_format", "{:.2f}".format)
    df = load()
    print("rows", len(df), "models", list(df.model.cat.categories))

    t = main_table(df, rng)
    show = ["model", "cond", "valid", "label_mass", "bal_acc", "spread", "rstd", "first_gain", "last_gain",
            "best_gain", "null_spread", "null_best_gain", "perm_p_spread", "flip_rate", "best_pos", "worst_pos",
            "top_pos", "cochran_p_holm", "chi2_p_holm"]
    print("\n=== MAIN TABLE ===\n", t[show].to_string(index=False))
    print("\n=== native-k inflation (points) ===\n",
          t[t.cond.isin(list(NATIVE))][["model", "cond", "valid", "bal_acc", "published_gain", "published_gain_lo",
                                        "published_gain_hi", "published_key_part", "published_key_part_lo",
                                        "published_key_part_hi", "published_order_part", "published_order_part_lo",
                                        "published_order_part_hi", "early2_gain", "early2_gain_lo", "early2_gain_hi",
                                        "best_gain", "best_gain_lo", "best_gain_hi"]].to_string(index=False))
    sh = [c for c in t.columns if c.startswith(("acc_", "share_")) and not c.endswith(("_lo", "_hi"))]
    print("\n=== acc / share by position ===\n", t[t.cond.isin(["mmlu_k4", "mmlu_k4_num", "csqa_k5", "boolq_ab", "boolq_word"])][["model", "cond"] + sh].to_string(index=False))

    pooled = pooled_by_cond(df, t, rng)
    keys = ["bal_acc", "spread", "rstd", "first_gain", "last_gain", "best_gain", "flip_rate", "early2_gain",
            "published_gain", "published_key_part", "published_order_part"]
    pp = pooled[["cond", "n_models"]].copy()
    for key in keys:
        pp[key] = [f"{a:.1f} [{b:.1f}, {c:.1f}]" if pd.notna(a) else "" for a, b, c in
                   zip(pooled.get(key), pooled.get(key + "_lo"), pooled.get(key + "_hi"))]
    print("\n=== POOLED BY CONDITION (mean of models) ===\n", pp.to_string(index=False))
    pp.to_csv(f"{EV}/pooled_by_condition_formatted.csv", index=False)
    print("\n=== FORMAT CONTRASTS ===\n", format_contrasts(df, t, rng).to_string(index=False))

    strata, dpool, dcmp = difficulty_tables(df, t, rng)
    dkeys = ["bal_acc", "spread", "directed_gap", "flip_rate", "best_gain", "published_gain", "attrib_share_best", "attrib_share_published"]
    dshow = dpool[["cond", "scheme", "nbin", "bin_name"]].copy()
    for key in dkeys:
        dshow[key] = [f"{a:.1f} [{b:.1f}, {c:.1f}]" for a, b, c in zip(dpool[key], dpool[key + "_lo"], dpool[key + "_hi"])]
    print("\n=== DIFFICULTY (mean of models) ===\n", dshow.to_string(index=False))
    dshow.to_csv(f"{EV}/difficulty_pooled_formatted.csv", index=False)
    print("\n=== DIFFICULTY CONTRASTS (terciles) ===\n", dcmp.to_string(index=False))
    mx = strata[(strata.nbin == 5)].sort_values("attrib_share_best", ascending=False).head(8)
    print("\n=== largest per-model stratum attributable shares (quintiles) ===\n",
          mx[["cond", "scheme", "bin_name", "model", "n_q", "bal_acc", "best_gain", "attrib_share_best", "spread"]].to_string(index=False))

    print("\n=== READ-OUT CHECK ===\n", readout_check(df).to_string(index=False))
    print("\n=== LOW LABEL MASS ROBUSTNESS ===\n", low_mass_robustness(df).to_string(index=False))
    print("\n=== SPLIT-HALF ===\n", subject_split(df).to_string(index=False))
    print("\n=== as-published gold distribution in our samples ===")
    one = df[(df.model == df.model.cat.categories[0]) & (df.rot == 0)]
    for c in ("mmlu_k4", "csqa_k5"):
        print(c, one[one.cond == c].orig_gold.value_counts(normalize=True).sort_index().round(3).to_dict())

    fig_acc_by_position(t)
    fig_selection_share(t)
    fig_format(t)
    fig_difficulty(dpool)
    fig_inflation(t)
    print("figures written")


if __name__ == "__main__":
    main()
