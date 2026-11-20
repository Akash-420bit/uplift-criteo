"""End-to-end uplift pipeline.

Usage (from the repository root):
    python -m src.pipeline --step all
    python -m src.pipeline --step prepare|train|evaluate|budget|report
    python -m src.pipeline --step all --synthetic 200000   # quick test, fake data

Each step reads what the previous step saved, so a run can be resumed after
an interruption. Trained-model predictions are cached in
<results>/predictions/ and are not recomputed if present.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
import warnings

import numpy as np
import pandas as pd
import yaml

from . import data as D
from . import metrics as M
from .models import METHODS, fit_and_score


def load_cfg(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def _norm_ci(diff, se):
    return diff - 1.96 * se, diff + 1.96 * se


def _ate(y, t):
    y1, y0 = y[t == 1], y[t == 0]
    d = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return d, se


def profile(df: pd.DataFrame) -> dict:
    t = df["treatment"].to_numpy()
    out = {"n_rows": len(df), "treatment_share": t.mean()}
    for y in ["visit", "conversion"]:
        v = df[y].to_numpy().astype(float)
        out[f"{y}_rate"] = v.mean()
        out[f"{y}_rate_treated"] = v[t == 1].mean()
        out[f"{y}_rate_control"] = v[t == 0].mean()
        d, se = _ate(v, t)
        lo, hi = _norm_ci(d, se)
        out[f"{y}_ate"], out[f"{y}_ate_ci_low"], out[f"{y}_ate_ci_high"] = d, lo, hi
        out[f"{y}_relative_lift"] = d / v[t == 0].mean()
    out["exposure_rate_treated"] = df["exposure"].to_numpy()[t == 1].mean()
    out["exposure_rate_control"] = df["exposure"].to_numpy()[t == 0].mean()
    return out


def randomization_check(df: pd.DataFrame) -> pd.DataFrame:
    t = df["treatment"].to_numpy() == 1
    rows = []
    for f in D.FEATURES:
        x = df[f].to_numpy().astype(float)
        m1, m0 = x[t].mean(), x[~t].mean()
        s1, s0 = x[t].std(ddof=1), x[~t].std(ddof=1)
        rows.append({"feature": f, "mean_treated": m1, "mean_control": m0,
                     "sd_treated": s1, "sd_control": s0,
                     "smd": (m1 - m0) / np.sqrt((s1 ** 2 + s0 ** 2) / 2)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Step 1: prepare
# ----------------------------------------------------------------------------
def step_prepare(cfg, args):
    seed = cfg["seed"]
    t0 = time.time()
    if args.synthetic:
        full = D.make_synthetic(args.synthetic, seed)
        n_sample = min(cfg["data"]["sample_size"], len(full) // 2)
    else:
        path = D.download(cfg, args.root) if not args.raw else args.raw
        print("Reading the full file (about 1-3 minutes) ...")
        full = D.load_full(path)
        n_sample = cfg["data"]["sample_size"]
    print(f"Full data: {len(full):,} rows")

    prof_full = profile(full)
    rc_full = randomization_check(full)
    df = D.simple_random_sample(full, n_sample, seed)
    del full
    print(f"Simple random sample: {len(df):,} rows (seed {seed})")
    sp = D.split(df, cfg, seed)

    prof_sample = profile(df)
    rows = [{"metric": k, "full_file": prof_full[k], "sample": prof_sample[k]} for k in prof_full]
    for s in ["train", "valid", "test"]:
        m = (sp == s).to_numpy()
        rows.append({"metric": f"n_{s}", "full_file": np.nan, "sample": int(m.sum())})
        rows.append({"metric": f"treatment_share_{s}", "full_file": np.nan,
                     "sample": df["treatment"].to_numpy()[m].mean()})
        rows.append({"metric": f"visit_rate_{s}", "full_file": np.nan,
                     "sample": df["visit"].to_numpy()[m].mean()})
    pd.DataFrame(rows).to_csv(os.path.join(args.results, "dataset_profile.csv"), index=False)

    rc = randomization_check(df)
    rc["smd_full_file"] = rc_full["smd"].to_numpy()
    rc.to_csv(os.path.join(args.results, "randomization_check.csv"), index=False)

    os.makedirs(args.work, exist_ok=True)
    np.savez_compressed(os.path.join(args.work, "sample_split.npz"),
                        X=df[D.FEATURES].to_numpy(np.float32),
                        treatment=df["treatment"].to_numpy(np.int8),
                        visit=df["visit"].to_numpy(np.int8),
                        conversion=df["conversion"].to_numpy(np.int8),
                        exposure=df["exposure"].to_numpy(np.int8),
                        split=sp.map({"train": 0, "valid": 1, "test": 2}).to_numpy(np.int8))
    print(f"prepare done in {time.time() - t0:.0f}s")


def _load_sample(args):
    z = np.load(os.path.join(args.work, "sample_split.npz"))
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------
# Step 2: train (one prediction file per outcome x method; skipped if present)
# ----------------------------------------------------------------------------
def step_train(cfg, args):
    d = _load_sample(args)
    X, T, sp = d["X"], d["treatment"], d["split"]
    tr, va, te = sp == 0, sp == 1, sp == 2
    pred_dir = os.path.join(args.results, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    infos = []
    for outcome in [cfg["outcomes"]["primary"], cfg["outcomes"]["secondary"]]:
        Y = d[outcome]
        for method in args.methods:
            f = os.path.join(pred_dir, f"{outcome}__{method}.npz")
            if os.path.exists(f):
                print(f"[skip] {outcome}/{method} already trained")
                infos.append(json.loads(str(np.load(f)["info"])))
                continue
            print(f"Training {method} for {outcome} ...", flush=True)
            (s_va, s_te), info = fit_and_score(method, X[tr], Y[tr], T[tr], X[va], Y[va], T[va],
                                               [X[va], X[te]], cfg, cfg["seed"])
            info["outcome"] = outcome
            np.savez_compressed(f, valid=s_va.astype(np.float64), test=s_te.astype(np.float64),
                                info=json.dumps(info, default=str))
            print(f"   done: {info}", flush=True)
            infos.append(info)
    pd.DataFrame(infos).to_csv(os.path.join(args.results, "model_info.csv"), index=False)


def _scores(args, outcome, method, part):
    return np.load(os.path.join(args.results, "predictions", f"{outcome}__{method}.npz"))[part]


# ----------------------------------------------------------------------------
# Step 3: evaluate on TEST (point estimates, bootstrap CIs, paired differences)
# ----------------------------------------------------------------------------
def step_evaluate(cfg, args):
    d = _load_sample(args)
    te = d["split"] == 2
    T = d["treatment"][te].astype(float)
    ev = cfg["evaluation"]
    top_k = ev["top_k"]
    rows, diff_rows, curve_rows = [], [], []
    for outcome in [cfg["outcomes"]["primary"], cfg["outcomes"]["secondary"]]:
        Y = d[outcome][te].astype(float)
        orders = {m: M.rank_order(_scores(args, outcome, m, "test"), cfg["seed"]) for m in args.methods}
        point = {m: M.summarize(Y[o], T[o], top_k=top_k) for m, o in orders.items()}
        # random targeting reference curve
        phi, qn_r, un_r = M.curve_points(Y, T, ev["curve_points"])
        for m, o in orders.items():
            phi, qn, un = M.curve_points(Y[o], T[o], ev["curve_points"])
            for p_, q_, u_ in zip(phi, qn, un):
                curve_rows.append({"outcome": outcome, "method": m, "phi": p_, "qini_per_user": q_,
                                   "uplift_curve_per_user": u_, "random_qini_per_user": p_ * qn[-1]})
        # paired bootstrap: same resampled users for every method
        boot = {m: [] for m in orders}
        t0 = time.time()
        for b, w in enumerate(M.bootstrap_weights(len(Y), ev["n_bootstrap"], cfg["seed"] + 7)):
            for m, o in orders.items():
                boot[m].append(M.summarize(Y[o], T[o], w=w[o], top_k=top_k))
            if (b + 1) % 50 == 0:
                print(f"  {outcome}: bootstrap {b + 1}/{ev['n_bootstrap']} ({time.time() - t0:.0f}s)", flush=True)
        keys = list(point["response"].keys())
        for m in orders:
            bdf = pd.DataFrame(boot[m])
            for k in keys:
                lo, hi = M.ci(bdf[k])
                rows.append({"outcome": outcome, "method": m, "metric": k, "estimate": point[m][k],
                             "ci_low": lo, "ci_high": hi, "boot_se": bdf[k].std(ddof=1)})
            if m == "response":
                continue
            base = pd.DataFrame(boot["response"])
            for k in keys:
                if k == "ate":
                    continue
                diffs = bdf[k].to_numpy() - base[k].to_numpy()
                lo, hi = M.ci(diffs)
                diff_rows.append({"outcome": outcome, "method": m, "metric": k,
                                  "difference_vs_response": point[m][k] - point["response"][k],
                                  "ci_low": lo, "ci_high": hi,
                                  "share_boot_le_0": float((diffs <= 0).mean())})
    corr_rows = []
    for outcome in [cfg["outcomes"]["primary"], cfg["outcomes"]["secondary"]]:
        sc = pd.DataFrame({m: _scores(args, outcome, m, "test") for m in args.methods})
        cm = sc.rank().corr()                       # Spearman rank correlation
        for a in args.methods:
            for b in args.methods:
                corr_rows.append({"outcome": outcome, "method_a": a, "method_b": b, "spearman": cm.loc[a, b]})
    pd.DataFrame(corr_rows).to_csv(os.path.join(args.results, "score_correlations.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(args.results, "main_results.csv"), index=False)
    pd.DataFrame(diff_rows).to_csv(os.path.join(args.results, "paired_differences.csv"), index=False)
    pd.DataFrame(curve_rows).to_csv(os.path.join(args.results, "qini_curves.csv"), index=False)
    print("evaluate done")


# ----------------------------------------------------------------------------
# Step 4: budget simulation (phi chosen on VALIDATION, profit reported on TEST)
# ----------------------------------------------------------------------------
def _best_phi(prof, phi_grid):
    return float(phi_grid[int(np.argmax(prof))])


def step_budget(cfg, args):
    d = _load_sample(args)
    outcome = cfg["outcomes"]["primary"]
    b = cfg["budget"]
    phi_grid = np.round(np.arange(0, 1 + 1e-9, b["phi_grid_step"]), 4)
    parts = {}
    for name, code in [("valid", 1), ("test", 2)]:
        m = d["split"] == code
        parts[name] = (d[outcome][m].astype(float), d["treatment"][m].astype(float))

    def sorted_part(method, part):
        y, t = parts[part]
        o = M.rank_order(_scores(args, outcome, method, part), cfg["seed"])
        return y[o], t[o], o

    def evaluate(V, c, with_ci=False):
        rows, curves, boot_profit = [], [], {}
        yv, tv = parts["valid"]
        yt, tt = parts["test"]
        # contact nobody / everybody / random targeting (expected profit is linear in phi)
        all_val = M.profit_curve(yv, tv, V, c, np.array([1.0]))[0]
        all_test = M.profit_curve(yt, tt, V, c, np.array([1.0]))[0]
        rows.append({"policy": "contact_none", "phi_selected": 0.0, "profit_valid": 0.0, "profit_test": 0.0})
        rows.append({"policy": "contact_all", "phi_selected": 1.0, "profit_valid": all_val, "profit_test": all_test})
        phi_r = 1.0 if all_val > 0 else 0.0
        rows.append({"policy": "random", "phi_selected": phi_r, "profit_valid": phi_r * all_val,
                     "profit_test": phi_r * all_test})
        for m in args.methods:
            ys, ts, _ = sorted_part(m, "valid")
            pv = M.profit_curve(ys, ts, V, c, phi_grid)
            phi_star = _best_phi(pv, phi_grid)
            ys2, ts2, o2 = sorted_part(m, "test")
            pt = M.profit_curve(ys2, ts2, V, c, phi_grid)
            k = int(np.where(phi_grid == phi_star)[0][0])
            row = {"policy": m, "phi_selected": phi_star, "profit_valid": pv[k], "profit_test": pt[k],
                   "phi_test_oracle": _best_phi(pt, phi_grid), "profit_test_oracle": pt.max()}
            if with_ci:
                # same seed for every method -> same resampled users (paired bootstrap)
                vals = np.array([M.profit_curve(ys2, ts2, V, c, np.array([phi_star]), w=w[o2])[0]
                                 for w in M.bootstrap_weights(len(ys2), cfg["evaluation"]["n_bootstrap"], cfg["seed"] + 11)])
                row["profit_test_ci_low"], row["profit_test_ci_high"] = M.ci(vals)
                boot_profit[m] = vals
            rows.append(row)
            for p_, a, bb in zip(phi_grid, pv, pt):
                curves.append({"method": m, "phi": p_, "profit_valid": a, "profit_test": bb})
        for p_ in phi_grid:
            curves.append({"method": "random", "phi": p_, "profit_valid": p_ * all_val, "profit_test": p_ * all_test})
        out = pd.DataFrame(rows)
        if with_ci:
            for m in args.methods[1:]:
                dlt = boot_profit[m] - boot_profit[args.methods[0]]
                i = out.index[out.policy == m][0]
                i0 = out.index[out.policy == args.methods[0]][0]
                out.loc[i, "diff_vs_response"] = out.loc[i, "profit_test"] - out.loc[i0, "profit_test"]
                out.loc[i, "diff_ci_low"], out.loc[i, "diff_ci_high"] = M.ci(dlt)
                out.loc[i, "diff_share_boot_le_0"] = float((dlt <= 0).mean())
        out.insert(0, "c", c)
        out.insert(0, "V", V)
        return out, pd.DataFrame(curves)

    main, curves = evaluate(b["V"], b["c"], with_ci=True)
    main.to_csv(os.path.join(args.results, "budget_simulation.csv"), index=False)
    curves.to_csv(os.path.join(args.results, "profit_curves.csv"), index=False)
    grid = pd.concat([evaluate(V, c)[0] for V in b["sensitivity"]["V"] for c in b["sensitivity"]["c"]])
    grid.to_csv(os.path.join(args.results, "sensitivity_grid.csv"), index=False)
    print("budget done")


# ----------------------------------------------------------------------------
# Step 5: figures and LaTeX tables
# ----------------------------------------------------------------------------
def step_report(cfg, args):
    from . import report
    report.make_all(cfg, args.results)
    import importlib.metadata as md
    env = {"python": platform.python_version()}
    for p in ["numpy", "pandas", "scikit-learn", "lightgbm", "econml", "matplotlib", "scipy", "PyYAML"]:
        try:
            env[p] = md.version(p)
        except md.PackageNotFoundError:
            env[p] = None
    with open(os.path.join(args.results, "run_environment.json"), "w") as fh:
        json.dump(env, fh, indent=2)
    print("report done")


STEPS = {"prepare": step_prepare, "train": step_train, "evaluate": step_evaluate,
         "budget": step_budget, "report": step_report}


def main():
    warnings.filterwarnings("ignore", message=".*eval_set.*")
    warnings.filterwarnings("ignore", category=FutureWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", default="all", choices=["all"] + list(STEPS))
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--root", default=".", help="folder where data/ is created")
    ap.add_argument("--raw", default=None, help="path to an already-downloaded raw .csv.gz")
    ap.add_argument("--results", default="results")
    ap.add_argument("--work", default=None, help="folder for the cached sample (default <root>/data)")
    ap.add_argument("--synthetic", type=int, default=0, help="use N synthetic rows instead of Criteo")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    if args.synthetic:
        cfg["data"]["sample_size"] = min(cfg["data"]["sample_size"], args.synthetic // 2)
        cfg["models"]["causal_forest"]["train_subsample"] = min(cfg["models"]["causal_forest"]["train_subsample"], 30000)
    args.work = args.work or os.path.join(args.root, "data")
    os.makedirs(args.results, exist_ok=True)
    steps = list(STEPS) if args.step == "all" else [args.step]
    for s in steps:
        print(f"\n===== step: {s} =====", flush=True)
        STEPS[s](cfg, args)


if __name__ == "__main__":
    main()
