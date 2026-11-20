"""Scoring methods compared in the paper (larger score = target first).

response       P(Y=1 | X), ignores treatment (common-practice baseline)
t_learner      mu1(X) - mu0(X), two separate classifiers
x_learner      Kuenzel et al. (2019): imputed effects, tau = p*tau0 + (1-p)*tau1
trans_out      regression on Z = Y*T/p - Y*(1-T)/(1-p)   (Athey & Imbens 2016)
dr_learner     Kennedy (2023): regression on the doubly robust pseudo-outcome,
               nuisance models cross-fitted (2 folds), known propensity p
causal_forest  EconML CausalForestDML (an R-learner forest), all training users
hybrid         rank blend of the response score and the best uplift score
               (built in pipeline.py, weight chosen on validation)

Treatment was randomized with a constant probability, so the propensity is the
treated share p of the training split (no propensity model is needed).
All LightGBM models use early stopping on the validation split.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
from sklearn.dummy import DummyClassifier

METHODS = ["response", "t_learner", "x_learner", "trans_out", "dr_learner", "causal_forest"]
ALL_METHODS = METHODS + ["hybrid"]
UPLIFT_METHODS = METHODS[1:]
SHORT = {"response": "Response", "t_learner": "T-learner", "x_learner": "X-learner",
         "trans_out": "Transf. outcome", "dr_learner": "DR-learner", "causal_forest": "Causal forest",
         "hybrid": "Hybrid"}


def _params(cfg, seed, override):
    p = dict(cfg["models"]["lightgbm"])
    p.pop("early_stopping_rounds")
    p.update(override or {})
    return dict(p, random_state=seed, n_jobs=cfg["models"]["n_jobs"], verbose=-1)


def _es(cfg):
    return [lgb.early_stopping(cfg["models"]["lightgbm"]["early_stopping_rounds"], verbose=False)]


def clf(Xtr, ytr, Xva, yva, cfg, seed, hp):
    m = lgb.LGBMClassifier(**_params(cfg, seed, hp))
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss", callbacks=_es(cfg))
    return m


def reg(Xtr, ztr, Xva, zva, cfg, seed, hp):
    m = lgb.LGBMRegressor(**_params(cfg, seed, hp))
    m.fit(Xtr, ztr, eval_set=[(Xva, zva)], eval_metric="l2", callbacks=_es(cfg))
    return m


def transformed_outcome(y, t, p):
    return y * t / p - y * (1 - t) / (1 - p)


def fit_and_score(method, tr, va, X_score_list, cfg, seed, hp=None):
    """tr, va: dicts with X, y, t. Returns (scores per X in X_score_list, info)."""
    t0 = time.time()
    Xtr, ytr, ttr = tr["X"], tr["y"], tr["t"]
    Xva, yva, tva = va["X"], va["y"], va["t"]
    p = float(ttr.mean())
    info = {"method": method, "hp": dict(hp or {}), "n_train": len(ytr)}
    if method == "response":
        m = clf(Xtr, ytr, Xva, yva, cfg, seed, hp)
        scores = [m.predict_proba(X)[:, 1] for X in X_score_list]
    elif method in ("t_learner", "x_learner"):
        m1 = clf(Xtr[ttr == 1], ytr[ttr == 1], Xva[tva == 1], yva[tva == 1], cfg, seed, hp)
        m0 = clf(Xtr[ttr == 0], ytr[ttr == 0], Xva[tva == 0], yva[tva == 0], cfg, seed + 1, hp)
        if method == "t_learner":
            scores = [m1.predict_proba(X)[:, 1] - m0.predict_proba(X)[:, 1] for X in X_score_list]
        else:
            def imputed(X, y, t):
                d1 = y[t == 1] - m0.predict_proba(X[t == 1])[:, 1]       # treated: Y - mu0
                d0 = m1.predict_proba(X[t == 0])[:, 1] - y[t == 0]       # control: mu1 - Y
                return d1, d0
            d1tr, d0tr = imputed(Xtr, ytr, ttr)
            d1va, d0va = imputed(Xva, yva, tva)
            g1 = reg(Xtr[ttr == 1], d1tr, Xva[tva == 1], d1va, cfg, seed + 2, hp)
            g0 = reg(Xtr[ttr == 0], d0tr, Xva[tva == 0], d0va, cfg, seed + 3, hp)
            scores = [p * g0.predict(X) + (1 - p) * g1.predict(X) for X in X_score_list]
    elif method == "trans_out":
        m = reg(Xtr, transformed_outcome(ytr, ttr, p), Xva, transformed_outcome(yva, tva, p), cfg, seed, hp)
        scores = [m.predict(X) for X in X_score_list]
    elif method == "dr_learner":
        rng = np.random.default_rng(seed)
        fold = rng.integers(0, 2, len(ytr))
        mu1_tr, mu0_tr = np.zeros(len(ytr)), np.zeros(len(ytr))
        mu1_va, mu0_va = np.zeros(len(yva)), np.zeros(len(yva))
        for k in (0, 1):
            fit = fold != k
            a1 = clf(Xtr[fit & (ttr == 1)], ytr[fit & (ttr == 1)], Xva[tva == 1], yva[tva == 1], cfg, seed + 10 + k, hp)
            a0 = clf(Xtr[fit & (ttr == 0)], ytr[fit & (ttr == 0)], Xva[tva == 0], yva[tva == 0], cfg, seed + 20 + k, hp)
            mu1_tr[~fit] = a1.predict_proba(Xtr[~fit])[:, 1]
            mu0_tr[~fit] = a0.predict_proba(Xtr[~fit])[:, 1]
            mu1_va += a1.predict_proba(Xva)[:, 1] / 2
            mu0_va += a0.predict_proba(Xva)[:, 1] / 2

        def pseudo(y, t, mu1, mu0):
            return mu1 - mu0 + t * (y - mu1) / p - (1 - t) * (y - mu0) / (1 - p)
        m = reg(Xtr, pseudo(ytr, ttr, mu1_tr, mu0_tr), Xva, pseudo(yva, tva, mu1_va, mu0_va), cfg, seed, hp)
        scores = [m.predict(X) for X in X_score_list]
    elif method == "causal_forest":
        from econml.dml import CausalForestDML
        c = dict(cfg["models"]["causal_forest"])
        c.update(hp or {})
        model_y = lgb.LGBMRegressor(**_params(cfg, seed, {"n_estimators": c["nuisance_n_estimators"]}))
        cf = CausalForestDML(model_y=model_y, model_t=DummyClassifier(strategy="prior"),
                             discrete_treatment=True, n_estimators=c["n_estimators"],
                             min_samples_leaf=c["min_samples_leaf"], max_samples=c["max_samples"],
                             cv=c["cv"], random_state=seed, n_jobs=cfg["models"]["n_jobs"])
        cf.fit(ytr.astype(float), ttr, X=Xtr)
        scores = [np.asarray(cf.effect(X)).ravel() for X in X_score_list]
    else:
        raise ValueError(method)
    info["fit_seconds"] = round(time.time() - t0, 1)
    return scores, info
