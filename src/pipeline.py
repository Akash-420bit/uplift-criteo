"""End-to-end uplift pipeline.

Usage (from the repository root):
    python -m src.pipeline --step all
    python -m src.pipeline --step prepare|train|evaluate|exposure|budget|seeds|report
    python -m src.pipeline --step all --synthetic 200000   # quick test, fake data

Each step reads what the previous step saved, so a run can be resumed after an
interruption. Model scores are cached in <results>/predictions/ and are not
recomputed if present.

Steps
  prepare   download + checksum, full-file profile, data-quality, imbalance and
            exposure (complier) analysis, 2M sample, duplicate-aware split
  train     every method x hyperparameter setting; the setting with the best
            validation AUQC is kept; hybrid score built on validation
  evaluate  test-set Qini/AUQC/top-k, 1,000 paired bootstrap resamples,
            Holm-adjusted p-values vs the response model
  exposure  ITT, exposure rate and complier effect by score decile (test set)
  budget    profit simulation, phi chosen on validation, sensitivity grid
  seeds     whole pipeline (new sample, split, tuning) for extra seeds, visit
  report    figures and LaTeX tables
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import time
import warnings

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from . import data as D
from . import metrics as M
from .models import ALL_METHODS, METHODS, UPLIFT_METHODS, fit_and_score


def load_cfg(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def _ate(y, t):
    y1, y0 = y[t == 1], y[t == 0]
    d = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return d, se


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def profile(df: pd.DataFrame) -> dict:
    t = df["treatment"].to_numpy()
    out = {"n_rows": len(df), "treatment_share": t.mean()}
    for y in ["visit", "conversion"]:
        v = df[y].to_numpy().astype(float)
        out[f"{y}_rate"] = v.mean()
        out[f"{y}_rate_treated"] = v[t == 1].mean()
        out[f"{y}_rate_control"] = v[t == 0].mean()
        d, se = _ate(v, t)
        out[f"{y}_ate"], out[f"{y}_ate_ci_low"], out[f"{y}_ate_ci_high"] = d, d - 1.96 * se, d + 1.96 * se
        out[f"{y}_relative_lift"] = d / v[t == 0].mean()
    out["exposure_rate_treated"] = df["exposure"].to_numpy()[t == 1].mean()
    out["exposure_rate_control"] = df["exposure"].to_numpy()[t == 0].mean()
    return out


def randomization_check(df: pd.DataFrame) -> pd.DataFrame:
    t = df["treatment"].to_numpy() == 1
    n1, n0 = t.sum(), (~t).sum()
    rows = []
    for f in D.FEATURES:
        x = df[f].to_numpy().astype(float)
        m1, m0 = x[t].mean(), x[~t].mean()
        s1, s0 = x[t].std(ddof=1), x[~t].std(ddof=1)
        z = (m1 - m0) / np.sqrt(s1 ** 2 / n1 + s0 ** 2 / n0)
        rows.append({"feature": f, "mean_treated": m1, "mean_control": m0, "sd_treated": s1, "sd_control": s0,
                     "smd": (m1 - m0) / np.sqrt((s1 ** 2 + s0 ** 2) / 2), "z": z,
                     "p_value": 2 * stats.norm.sf(abs(z))})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p_value)
    out["sd_smd_under_randomization"] = np.sqrt(1 / n1 + 1 / n0)
    return out


# ----------------------------------------------------------------------------- full-file analyses
def exposure_full(df: pd.DataFrame) -> pd.DataFrame:
    """Complier average causal effect (CACE) with treatment as the instrument.

    One-sided non-compliance (control users cannot be exposed), so under the
    exclusion restriction (an ad that is never shown has no effect)
    CACE = ITT / P(exposed | treated)  (Bloom 1984; Angrist, Imbens & Rubin 1996).
    Standard error by the delta method (covariance included).
    """
    t = df["treatment"].to_numpy() == 1
    e = df["exposure"].to_numpy().astype(float)
    pi = e[t].mean()
    rows = []
    for y in ["visit", "conversion"]:
        v = df[y].to_numpy().astype(float)
        itt, itt_se = _ate(v, t.astype(int))
        cace = itt / pi
        var = (np.var(v[t] - cace * e[t], ddof=1) / t.sum() + np.var(v[~t], ddof=1) / (~t).sum()) / pi ** 2
        se = np.sqrt(var)
        rows.append({"outcome": y, "n_treated": int(t.sum()), "n_control": int((~t).sum()),
                     "exposure_rate_treated": pi, "exposure_rate_control": e[~t].mean(),
                     "itt": itt, "itt_ci_low": itt - 1.96 * itt_se, "itt_ci_high": itt + 1.96 * itt_se,
                     "cace": cace, "cace_ci_low": cace - 1.96 * se, "cace_ci_high": cace + 1.96 * se,
                     "outcome_rate_exposed_treated": v[t & (e == 1)].mean(),
                     "outcome_rate_unexposed_treated": v[t & (e == 0)].mean(),
                     "outcome_rate_control": v[~t].mean()})
    return pd.DataFrame(rows)


def adjusted_ate(df: pd.DataFrame) -> list:
    """Unadjusted vs regression-adjusted ATE (OLS on T and standardized features, HC1 SE)."""
    X = df[D.FEATURES].to_numpy(np.float64)
    X = (X - X.mean(0)) / X.std(0)
    t = df["treatment"].to_numpy(np.float64)
    Z = np.column_stack([np.ones(len(t)), t, X])
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    rows = []
    for y in ["visit", "conversion"]:
        v = df[y].to_numpy(np.float64)
        beta = ZtZ_inv @ (Z.T @ v)
        resid = v - Z @ beta
        meat = (Z * resid[:, None] ** 2).T @ Z
        cov = ZtZ_inv @ meat @ ZtZ_inv * len(v) / (len(v) - Z.shape[1])
        d, se = _ate(v, t)
        rows += [{"analysis": f"{y}_ate_unadjusted", "value": d, "se": se},
                 {"analysis": f"{y}_ate_regression_adjusted", "value": beta[1], "se": np.sqrt(cov[1, 1])}]
        del meat
    return rows


def treatment_classifier_auc(d: dict, cfg: dict, seed: int) -> list:
    """Two-sample test: can the features predict treatment? AUC 0.5 under perfect randomization."""
    from sklearn.metrics import roc_auc_score
    from .models import clf
    tr, va, te = d["split"] == 0, d["split"] == 1, d["split"] == 2
    m = clf(d["X"][tr], d["treatment"][tr], d["X"][va], d["treatment"][va], cfg, seed, {})
    s = m.predict_proba(d["X"][te])[:, 1]
    t = d["treatment"][te]
    auc = roc_auc_score(t, s)
    rng = np.random.default_rng(seed + 5)
    boots = []
    for _ in range(200):
        i = rng.integers(0, len(t), len(t))
        boots.append(roc_auc_score(t[i], s[i]))
    lo, hi = M.ci(boots)
    return [{"analysis": "treatment_classifier_auc_test", "value": auc, "se": float(np.std(boots, ddof=1)),
             "ci_low": lo, "ci_high": hi}]


# ----------------------------------------------------------------------------- data loading helpers
def load_full_data(cfg, args):
    if args.synthetic:
        return D.make_synthetic(args.synthetic, cfg["seed"])
    path = args.raw or D.download(cfg, args.root)
    print("Reading the full file (about 1-3 minutes) ...", flush=True)
    return D.load_full(path)


def sample_and_split(full, cfg, seed, n_sample):
    df = D.simple_random_sample(full, n_sample, seed)
    sp = D.split(df, cfg, seed)
    d = {"X": df[D.FEATURES].to_numpy(np.float32), "treatment": df["treatment"].to_numpy(np.int8),
         "visit": df["visit"].to_numpy(np.int8), "conversion": df["conversion"].to_numpy(np.int8),
         "exposure": df["exposure"].to_numpy(np.int8),
         "split": sp.map({"train": 0, "valid": 1, "test": 2}).to_numpy(np.int8)}
    return df, d


# ----------------------------------------------------------------------------- step 1: prepare
def step_prepare(cfg, args):
    seed, t0 = cfg["seed"], time.time()
    full = load_full_data(cfg, args)
    print(f"Full data: {len(full):,} rows", flush=True)
    n_sample = args.n_sample

    prof_full = profile(full)
    rc_full = randomization_check(full)
    exposure_full(full).to_csv(os.path.join(args.results, "exposure_full.csv"), index=False)
    imb = adjusted_ate(full)

    key = pd.util.hash_pandas_object(full, index=False).to_numpy()
    keyx = pd.util.hash_pandas_object(full[D.FEATURES], index=False).to_numpy()
    quality = [{"metric": "rows_full_file", "value": len(full)},
               {"metric": "exact_duplicate_rows_full_file", "value": len(key) - len(np.unique(key))},
               {"metric": "duplicate_feature_vectors_full_file", "value": len(keyx) - len(np.unique(keyx))}]
    del key, keyx

    df, d = sample_and_split(full, cfg, seed, n_sample)
    del full
    print(f"Simple random sample: {len(df):,} rows (seed {seed})", flush=True)

    # leakage check: share of test rows whose feature vector also occurs in train
    hx = pd.util.hash_pandas_object(df[D.FEATURES], index=False).to_numpy()
    quality.append({"metric": "test_rows_with_feature_vector_in_train",
                    "value": float(np.isin(hx[d["split"] == 2], hx[d["split"] == 0]).mean())})
    from sklearn.model_selection import train_test_split
    strata = d["treatment"].astype(int) * 2 + d["visit"].astype(int)
    a, b = train_test_split(np.arange(len(hx)), train_size=0.6, stratify=strata, random_state=seed)
    quality.append({"metric": "same_share_if_split_ignored_duplicates",
                    "value": float(np.isin(hx[b], hx[a]).mean())})
    pd.DataFrame(quality).to_csv(os.path.join(args.results, "data_quality.csv"), index=False)

    prof_sample = profile(df)
    rows = [{"metric": k, "full_file": prof_full[k], "sample": prof_sample[k]} for k in prof_full]
    for s, code in [("train", 0), ("valid", 1), ("test", 2)]:
        m = d["split"] == code
        rows += [{"metric": f"n_{s}", "full_file": np.nan, "sample": int(m.sum())},
                 {"metric": f"treatment_share_{s}", "full_file": np.nan, "sample": d["treatment"][m].mean()},
                 {"metric": f"visit_rate_{s}", "full_file": np.nan, "sample": d["visit"][m].mean()}]
    pd.DataFrame(rows).to_csv(os.path.join(args.results, "dataset_profile.csv"), index=False)

    rc = randomization_check(df)
    for col in ["smd", "z", "p_value", "p_holm", "sd_smd_under_randomization"]:
        rc[f"{col}_full_file"] = rc_full[col].to_numpy()
    rc.to_csv(os.path.join(args.results, "randomization_check.csv"), index=False)

    imb += treatment_classifier_auc(d, cfg, seed)
    pd.DataFrame(imb).to_csv(os.path.join(args.results, "imbalance_tests.csv"), index=False)

    os.makedirs(args.work, exist_ok=True)
    np.savez_compressed(os.path.join(args.work, "sample_split.npz"), **d)
    print(f"prepare done in {time.time() - t0:.0f}s", flush=True)


def _load_sample(args):
    z = np.load(os.path.join(args.work, "sample_split.npz"))
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- training with tuning
def val_auqc(score, y, t, seed):
    o = M.rank_order(score, seed)
    return M.summarize(y[o].astype(float), t[o].astype(float))["auqc"]


def pct_rank(s):
    return pd.Series(s).rank(pct=True, method="average").to_numpy()


def train_all(d, outcome, cfg, seed, pred_dir, methods=METHODS):
    """Tune and fit every method; returns {method: (valid_scores, test_scores)} and tuning rows."""
    os.makedirs(pred_dir, exist_ok=True)
    X, T, Y, sp = d["X"], d["treatment"], d[outcome], d["split"]
    part = {k: {"X": X[sp == c], "y": Y[sp == c], "t": T[sp == c]} for k, c in [("tr", 0), ("va", 1), ("te", 2)]}
    yv, tv = part["va"]["y"], part["va"]["t"]
    yt, tt = part["te"]["y"], part["te"]["t"]          # test AUQC recorded for reporting ONLY, never for selection
    best, tuning = {}, []
    for method in methods:
        grid = cfg["models"]["tuning_grid"][method]
        keys = sorted(grid)
        configs = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
        cand = []
        for k, hp in enumerate(configs):
            f = os.path.join(pred_dir, f"{outcome}__{method}__cfg{k}.npz")
            if os.path.exists(f):
                z = np.load(f)
                s_va, s_te, info = z["valid"], z["test"], json.loads(str(z["info"]))
            else:
                print(f"  [{outcome}] {method} cfg{k} {hp} ...", flush=True)
                (s_va, s_te), info = fit_and_score(method, part["tr"], part["va"], [part["va"]["X"], part["te"]["X"]],
                                                   cfg, seed, hp)
                np.savez_compressed(f, valid=s_va.astype(np.float64), test=s_te.astype(np.float64),
                                    info=json.dumps(info, default=str))
            a = val_auqc(s_va, yv, tv, seed)
            cand.append((a, k, s_va, s_te))
            tuning.append({"outcome": outcome, "seed": seed, "method": method, "config": k,
                           "hyperparameters": json.dumps(hp), "valid_auqc": a,
                           "test_auqc_report_only": val_auqc(s_te, yt, tt, seed),
                           "fit_seconds": info.get("fit_seconds")})
        a, k, s_va, s_te = max(cand, key=lambda c: c[0])
        best[method] = (s_va, s_te)
        for r in tuning:
            if r["method"] == method and r["outcome"] == outcome and r["seed"] == seed:
                r["selected"] = r["config"] == k
    # hybrid: blend of response rank and the best uplift rank (by validation AUQC)
    up = max((m for m in methods if m in UPLIFT_METHODS), key=lambda m: val_auqc(best[m][0], yv, tv, seed))
    rv, rt = [pct_rank(best["response"][i]) for i in (0, 1)], [pct_rank(best[up][i]) for i in (0, 1)]
    cand = []
    for w in cfg["models"]["hybrid_weights"]:
        s_va = w * rv[0] + (1 - w) * rt[0]
        cand.append((val_auqc(s_va, yv, tv, seed), w))
        tuning.append({"outcome": outcome, "seed": seed, "method": "hybrid", "config": f"w={w}",
                       "hyperparameters": json.dumps({"uplift_component": up, "weight_response": w}),
                       "valid_auqc": cand[-1][0],
                       "test_auqc_report_only": val_auqc(w * rv[1] + (1 - w) * rt[1], yt, tt, seed),
                       "fit_seconds": 0.0})
    a, w = max(cand)
    for r in tuning:
        if r["method"] == "hybrid" and r["outcome"] == outcome and r["seed"] == seed:
            r["selected"] = r["config"] == f"w={w}"
    best["hybrid"] = (w * rv[0] + (1 - w) * rt[0], w * rv[1] + (1 - w) * rt[1])
    return best, tuning


def step_train(cfg, args):
    d = _load_sample(args)
    pred_dir = os.path.join(args.results, "predictions")
    all_tuning = []
    for outcome in [cfg["outcomes"]["primary"], cfg["outcomes"]["secondary"]]:
        best, tuning = train_all(d, outcome, cfg, cfg["seed"], pred_dir)
        for m, (s_va, s_te) in best.items():
            np.savez_compressed(os.path.join(pred_dir, f"{outcome}__{m}.npz"), valid=s_va, test=s_te)
        all_tuning += tuning
    pd.DataFrame(all_tuning).to_csv(os.path.join(args.results, "tuning.csv"), index=False)
    print("train done", flush=True)


def _scores(args, outcome, method, part):
    return np.load(os.path.join(args.results, "predictions", f"{outcome}__{method}.npz"))[part]


# ----------------------------------------------------------------------------- step 3: evaluate
def step_evaluate(cfg, args):
    d = _load_sample(args)
    te = d["split"] == 2
    T = d["treatment"][te].astype(float)
    ev, top_k, B = cfg["evaluation"], cfg["evaluation"]["top_k"], cfg["evaluation"]["n_bootstrap"]
    rows, diff_rows, curve_rows, corr_rows = [], [], [], []
    for outcome in [cfg["outcomes"]["primary"], cfg["outcomes"]["secondary"]]:
        Y = d[outcome][te].astype(float)
        orders = {m: M.rank_order(_scores(args, outcome, m, "test"), cfg["seed"]) for m in ALL_METHODS}
        point = {m: M.summarize(Y[o], T[o], top_k=top_k) for m, o in orders.items()}
        for m, o in orders.items():
            phi, qn, un = M.curve_points(Y[o], T[o], ev["curve_points"])
            for p_, q_, u_ in zip(phi, qn, un):
                curve_rows.append({"outcome": outcome, "method": m, "phi": p_, "qini_per_user": q_,
                                   "uplift_curve_per_user": u_, "random_qini_per_user": p_ * qn[-1]})
        boot = {m: [] for m in orders}
        t0 = time.time()
        for b, w in enumerate(M.bootstrap_weights(len(Y), B, cfg["seed"] + 7)):
            for m, o in orders.items():
                boot[m].append(M.summarize(Y[o], T[o], w=w[o], top_k=top_k))
            if (b + 1) % 250 == 0:
                print(f"  {outcome}: bootstrap {b + 1}/{B} ({time.time() - t0:.0f}s)", flush=True)
        boot = {m: pd.DataFrame(v) for m, v in boot.items()}
        keys = list(point["response"].keys())
        for m in orders:
            for k in keys:
                lo, hi = M.ci(boot[m][k])
                rows.append({"outcome": outcome, "method": m, "metric": k, "estimate": point[m][k],
                             "ci_low": lo, "ci_high": hi, "boot_se": boot[m][k].std(ddof=1)})
        fam = []
        for m in ALL_METHODS[1:]:
            for k in keys:
                if k == "ate":
                    continue
                dl = boot[m][k].to_numpy() - boot["response"][k].to_numpy()
                lo, hi = M.ci(dl)
                p = max(min(1.0, 2 * min((dl <= 0).mean(), (dl >= 0).mean())), 1 / B)
                fam.append({"outcome": outcome, "method": m, "metric": k,
                            "difference_vs_response": point[m][k] - point["response"][k],
                            "ci_low": lo, "ci_high": hi, "share_boot_le_0": float((dl <= 0).mean()),
                            "p_value": p})
        adj = holm([r["p_value"] for r in fam])
        for r, a in zip(fam, adj):
            r["p_holm"] = a
            r["significant_holm_5pct"] = bool(a < 0.05)
        diff_rows += fam
        sc = pd.DataFrame({m: _scores(args, outcome, m, "test") for m in ALL_METHODS}).rank().corr()
        for a_ in ALL_METHODS:
            for b_ in ALL_METHODS:
                corr_rows.append({"outcome": outcome, "method_a": a_, "method_b": b_, "spearman": sc.loc[a_, b_]})
    pd.DataFrame(rows).to_csv(os.path.join(args.results, "main_results.csv"), index=False)
    pd.DataFrame(diff_rows).to_csv(os.path.join(args.results, "paired_differences.csv"), index=False)
    pd.DataFrame(curve_rows).to_csv(os.path.join(args.results, "qini_curves.csv"), index=False)
    pd.DataFrame(corr_rows).to_csv(os.path.join(args.results, "score_correlations.csv"), index=False)
    print("evaluate done", flush=True)


# ----------------------------------------------------------------------------- step 4: exposure by score decile
def step_exposure(cfg, args):
    """Is the ranking driven by who gets to SEE an ad? ITT, exposure rate among
    treated, and complier effect (ITT / exposure rate) by score decile, test set."""
    d = _load_sample(args)
    te = d["split"] == 2
    T, E, Y = d["treatment"][te], d["exposure"][te].astype(float), d["visit"][te].astype(float)
    rows = []
    for m in ALL_METHODS:
        s = _scores(args, "visit", m, "test")
        o = M.rank_order(s, cfg["seed"])
        dec = np.empty(len(s), dtype=int)
        dec[o] = np.arange(len(s)) * 10 // len(s) + 1          # decile 1 = highest score
        for k in range(1, 11):
            g = dec == k
            t1, t0 = g & (T == 1), g & (T == 0)
            itt = Y[t1].mean() - Y[t0].mean()
            pi = E[t1].mean()
            rows.append({"method": m, "decile": k, "n": int(g.sum()), "itt": itt, "exposure_rate_treated": pi,
                         "cace": itt / pi if pi > 0 else np.nan,
                         "visit_rate_exposed_treated": Y[t1 & (E == 1)].mean(),
                         "visit_rate_unexposed_treated": Y[t1 & (E == 0)].mean(),
                         "visit_rate_control": Y[t0].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.results, "exposure_by_decile.csv"), index=False)
    # how much of each ranking is "exposure ranking"? correlation over deciles
    summ = []
    for m, g in out.groupby("method", sort=False):
        summ.append({"method": m, "exposure_rate_top_decile": g.exposure_rate_treated.iloc[0],
                     "exposure_rate_bottom_decile": g.exposure_rate_treated.iloc[-1],
                     "spearman_itt_vs_exposure_over_deciles": stats.spearmanr(g.itt, g.exposure_rate_treated)[0],
                     "cace_top_decile": g.cace.iloc[0], "cace_deciles_2_to_10_mean": g.cace.iloc[1:].mean()})
    pd.DataFrame(summ).to_csv(os.path.join(args.results, "exposure_summary.csv"), index=False)
    print("exposure done", flush=True)


# ----------------------------------------------------------------------------- step 5: budget
def _profit_on_split(y, t, score, V, c, phi_grid, seed):
    o = M.rank_order(score, seed)
    return M.profit_curve(y[o], t[o], V, c, phi_grid), o


def step_budget(cfg, args):
    d = _load_sample(args)
    outcome = cfg["outcomes"]["primary"]
    b = cfg["budget"]
    phi_grid = np.round(np.arange(0, 1 + 1e-9, b["phi_grid_step"]), 4)
    parts = {}
    for name, code in [("valid", 1), ("test", 2)]:
        m = d["split"] == code
        parts[name] = (d[outcome][m].astype(float), d["treatment"][m].astype(float))

    def evaluate(V, c, with_ci=False):
        rows, curves, boot_profit = [], [], {}
        yv, tv = parts["valid"]
        yt, tt = parts["test"]
        all_val = M.profit_curve(yv, tv, V, c, np.array([1.0]))[0]
        all_test = M.profit_curve(yt, tt, V, c, np.array([1.0]))[0]
        rows.append({"policy": "contact_none", "phi_selected": 0.0, "profit_valid": 0.0, "profit_test": 0.0})
        rows.append({"policy": "contact_all", "phi_selected": 1.0, "profit_valid": all_val, "profit_test": all_test})
        phi_r = 1.0 if all_val > 0 else 0.0
        rows.append({"policy": "random", "phi_selected": phi_r, "profit_valid": phi_r * all_val,
                     "profit_test": phi_r * all_test})
        for m in ALL_METHODS:
            pv, _ = _profit_on_split(yv, tv, _scores(args, outcome, m, "valid"), V, c, phi_grid, cfg["seed"])
            k = int(np.argmax(pv))
            pt, o2 = _profit_on_split(yt, tt, _scores(args, outcome, m, "test"), V, c, phi_grid, cfg["seed"])
            row = {"policy": m, "phi_selected": float(phi_grid[k]), "profit_valid": pv[k], "profit_test": pt[k],
                   "phi_test_oracle": float(phi_grid[int(np.argmax(pt))]), "profit_test_oracle": pt.max()}
            if with_ci:
                ys2, ts2 = yt[o2], tt[o2]
                vals = np.array([M.profit_curve(ys2, ts2, V, c, np.array([phi_grid[k]]), w=w[o2])[0]
                                 for w in M.bootstrap_weights(len(ys2), cfg["evaluation"]["n_bootstrap"],
                                                              cfg["seed"] + 11)])
                row["profit_test_ci_low"], row["profit_test_ci_high"] = M.ci(vals)
                boot_profit[m] = vals
            rows.append(row)
            for p_, a, bb in zip(phi_grid, pv, pt):
                curves.append({"method": m, "phi": p_, "profit_valid": a, "profit_test": bb})
        for p_ in phi_grid:
            curves.append({"method": "random", "phi": p_, "profit_valid": p_ * all_val, "profit_test": p_ * all_test})
        out = pd.DataFrame(rows)
        if with_ci:
            i0 = out.index[out.policy == "response"][0]
            ps, idx = [], []
            for m in ALL_METHODS[1:]:
                dl = boot_profit[m] - boot_profit["response"]
                i = out.index[out.policy == m][0]
                out.loc[i, "diff_vs_response"] = out.loc[i, "profit_test"] - out.loc[i0, "profit_test"]
                out.loc[i, "diff_ci_low"], out.loc[i, "diff_ci_high"] = M.ci(dl)
                p = max(min(1.0, 2 * min((dl <= 0).mean(), (dl >= 0).mean())), 1 / len(dl))
                out.loc[i, "diff_p_value"] = p
                ps.append(p)
                idx.append(i)
            out.loc[idx, "diff_p_holm"] = holm(ps)
        out.insert(0, "c", c)
        out.insert(0, "V", V)
        return out, pd.DataFrame(curves)

    main, curves = evaluate(b["V"], b["c"], with_ci=True)
    main.to_csv(os.path.join(args.results, "budget_simulation.csv"), index=False)
    curves.to_csv(os.path.join(args.results, "profit_curves.csv"), index=False)
    grid = pd.concat([evaluate(V, c)[0] for V in b["sensitivity"]["V"] for c in b["sensitivity"]["c"]])
    grid.to_csv(os.path.join(args.results, "sensitivity_grid.csv"), index=False)
    print("budget done", flush=True)


# ----------------------------------------------------------------------------- step 6: seeds
def _seed_metrics(d, scores, cfg, seed):
    te, va = d["split"] == 2, d["split"] == 1
    Yt, Tt = d["visit"][te].astype(float), d["treatment"][te].astype(float)
    Yv, Tv = d["visit"][va].astype(float), d["treatment"][va].astype(float)
    b = cfg["budget"]
    phi_grid = np.round(np.arange(0, 1 + 1e-9, b["phi_grid_step"]), 4)
    rows = []
    for m, (s_va, s_te) in scores.items():
        o = M.rank_order(s_te, seed)
        r = M.summarize(Yt[o], Tt[o], top_k=cfg["evaluation"]["top_k"])
        pv, _ = _profit_on_split(Yv, Tv, s_va, b["V"], b["c"], phi_grid, seed)
        pt, _ = _profit_on_split(Yt, Tt, s_te, b["V"], b["c"], phi_grid, seed)
        k = int(np.argmax(pv))
        r.update({"seed": seed, "method": m, "phi_selected": float(phi_grid[k]), "profit_test": pt[k]})
        rows.append(r)
    return rows


def step_seeds(cfg, args):
    rows, tuning = [], []
    d = _load_sample(args)
    scores = {m: (_scores(args, "visit", m, "valid"), _scores(args, "visit", m, "test")) for m in ALL_METHODS}
    rows += _seed_metrics(d, scores, cfg, cfg["seed"])
    full = None
    for seed in cfg["extra_seeds"]:
        pred_dir = os.path.join(args.results, "predictions", f"seed{seed}")
        cache = os.path.join(args.work, f"sample_split_seed{seed}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            d = {k: z[k] for k in z.files}
        else:
            if full is None:
                full = load_full_data(cfg, args)
            _, d = sample_and_split(full, cfg, seed, args.n_sample)
            np.savez_compressed(cache, **d)
        print(f"seed {seed}: training ...", flush=True)
        best, tun = train_all(d, "visit", cfg, seed, pred_dir)
        tuning += tun
        rows += _seed_metrics(d, best, cfg, seed)
        pd.DataFrame(rows).to_csv(os.path.join(args.results, "seed_results.csv"), index=False)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.results, "seed_results.csv"), index=False)
    pd.DataFrame(tuning).to_csv(os.path.join(args.results, "seed_tuning.csv"), index=False)
    base = res[res.method == "response"].set_index("seed")
    summ = []
    for m, g in res.groupby("method", sort=False):
        g = g.set_index("seed")
        rec = {"method": m, "n_seeds": len(g)}
        for k in ["auqc", "uplift_top10", "uplift_top20", "uplift_top30", "profit_test"]:
            rec[f"{k}_mean"], rec[f"{k}_sd"] = g[k].mean(), g[k].std(ddof=1)
            rec[f"{k}_min"], rec[f"{k}_max"] = g[k].min(), g[k].max()
            rec[f"{k}_seeds_better_than_response"] = int((g[k] > base[k]).sum()) if m != "response" else np.nan
        summ.append(rec)
    pd.DataFrame(summ).to_csv(os.path.join(args.results, "seed_summary.csv"), index=False)
    print("seeds done", flush=True)


# ----------------------------------------------------------------------------- step 7: report
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
    print("report done", flush=True)


STEPS = {"prepare": step_prepare, "train": step_train, "evaluate": step_evaluate, "exposure": step_exposure,
         "budget": step_budget, "seeds": step_seeds, "report": step_report}


def main():
    warnings.filterwarnings("ignore", message=".*eval_set.*")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", default="all", choices=["all"] + list(STEPS))
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--root", default=".", help="folder where data/ is created")
    ap.add_argument("--raw", default=None, help="path to an already-downloaded raw .csv.gz")
    ap.add_argument("--results", default="results")
    ap.add_argument("--work", default=None, help="folder for the cached samples (default <root>/data)")
    ap.add_argument("--synthetic", type=int, default=0, help="use N synthetic rows instead of Criteo")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    args.n_sample = cfg["data"]["sample_size"]
    if args.synthetic:
        args.n_sample = min(args.n_sample, args.synthetic // 2)
        cfg["evaluation"]["n_bootstrap"] = 200
    args.work = args.work or os.path.join(args.root, "data")
    os.makedirs(args.results, exist_ok=True)
    steps = list(STEPS) if args.step == "all" else [args.step]
    for s in steps:
        print(f"\n===== step: {s} =====", flush=True)
        STEPS[s](cfg, args)


if __name__ == "__main__":
    main()
