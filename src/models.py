"""The four scoring methods compared in the paper.

(a) response   : P(Y=1 | X), ignores treatment (common-practice baseline)
(b) t_learner  : P(Y=1 | X, T=1) - P(Y=1 | X, T=0), two separate models
(c) trans_out  : regression on the propensity-weighted transformed outcome
                 Z = Y*T/p - Y*(1-T)/(1-p)  (Athey & Imbens), E[Z|X] = tau(X)
(d) causal_forest : EconML CausalForestDML, trained on a documented subsample

All methods return a score where larger = target first.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
from sklearn.dummy import DummyClassifier

METHODS = ["response", "t_learner", "trans_out", "causal_forest"]
LABELS = {"response": "Response model (baseline)", "t_learner": "Two-model (T-learner)",
          "trans_out": "Transformed outcome", "causal_forest": "Causal forest"}
SHORT = {"response": "Response", "t_learner": "T-learner", "trans_out": "Transf. outcome",
         "causal_forest": "Causal forest"}


def _lgb_params(cfg: dict, seed: int, **override) -> dict:
    p = dict(cfg["models"]["lightgbm"])
    p.pop("early_stopping_rounds")
    p.update(override)
    return dict(p, random_state=seed, n_jobs=cfg["models"]["n_jobs"], verbose=-1)


def _fit_clf(Xtr, ytr, Xva, yva, cfg, seed, **override):
    m = lgb.LGBMClassifier(**_lgb_params(cfg, seed, **override))
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(cfg["models"]["lightgbm"]["early_stopping_rounds"], verbose=False)])
    return m


def _fit_reg(Xtr, ztr, Xva, zva, cfg, seed, **override):
    m = lgb.LGBMRegressor(**_lgb_params(cfg, seed, **override))
    m.fit(Xtr, ztr, eval_set=[(Xva, zva)], eval_metric="l2",
          callbacks=[lgb.early_stopping(cfg["models"]["lightgbm"]["early_stopping_rounds"], verbose=False)])
    return m


def transformed_outcome(y, t, p):
    return y * t / p - y * (1 - t) / (1 - p)


def fit_and_score(method, Xtr, ytr, ttr, Xva, yva, tva, X_score_list, cfg, seed):
    """Fit one method; return (list of score arrays for X_score_list, info dict)."""
    t0 = time.time()
    info = {"method": method}
    if method == "response":
        m = _fit_clf(Xtr, ytr, Xva, yva, cfg, seed)
        scores = [m.predict_proba(X)[:, 1] for X in X_score_list]
        info["n_trees"] = m.best_iteration_
    elif method == "t_learner":
        m1 = _fit_clf(Xtr[ttr == 1], ytr[ttr == 1], Xva[tva == 1], yva[tva == 1], cfg, seed)
        m0 = _fit_clf(Xtr[ttr == 0], ytr[ttr == 0], Xva[tva == 0], yva[tva == 0], cfg, seed + 1)
        scores = [m1.predict_proba(X)[:, 1] - m0.predict_proba(X)[:, 1] for X in X_score_list]
        info["n_trees"] = f"{m1.best_iteration_}/{m0.best_iteration_}"
    elif method == "trans_out":
        p = float(ttr.mean())                       # treatment share estimated on train
        info["p_treat"] = p
        ztr, zva = transformed_outcome(ytr, ttr, p), transformed_outcome(yva, tva, p)
        m = _fit_reg(Xtr, ztr, Xva, zva, cfg, seed, **cfg["models"]["transformed_outcome"])
        scores = [m.predict(X) for X in X_score_list]
        info["n_trees"] = m.best_iteration_
    elif method == "causal_forest":
        from econml.dml import CausalForestDML
        c = cfg["models"]["causal_forest"]
        rng = np.random.default_rng(seed)
        n_sub = min(c["train_subsample"], len(ytr))
        idx = rng.choice(len(ytr), size=n_sub, replace=False)
        model_y = lgb.LGBMRegressor(**_lgb_params(cfg, seed, n_estimators=c["nuisance_n_estimators"]))
        # Treatment is randomized with a constant probability, so the propensity
        # model is the (known) treatment share: a prior-only classifier.
        cf = CausalForestDML(model_y=model_y, model_t=DummyClassifier(strategy="prior"),
                             discrete_treatment=True, n_estimators=c["n_estimators"],
                             min_samples_leaf=c["min_samples_leaf"], max_samples=c["max_samples"],
                             cv=c["cv"], random_state=seed, n_jobs=cfg["models"]["n_jobs"])
        cf.fit(ytr[idx].astype(float), ttr[idx], X=Xtr[idx])
        scores = [np.asarray(cf.effect(X)).ravel() for X in X_score_list]
        info["n_train"] = n_sub
    else:
        raise ValueError(method)
    info.setdefault("n_train", len(ytr))
    info["fit_seconds"] = round(time.time() - t0, 1)
    return scores, info
