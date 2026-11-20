"""Figures (PDF + PNG, IEEE single-column width) and booktabs LaTeX tables,
generated only from the CSV files in results/."""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .models import ALL_METHODS, SHORT  # noqa: E402

# validated categorical palette (fixed order); lines also differ by style/marker
COLORS = {"response": "#2a78d6", "t_learner": "#eb6834", "x_learner": "#1baf7a", "trans_out": "#eda100",
          "dr_learner": "#e87ba4", "causal_forest": "#008300", "hybrid": "#6250d6", "random": "#8a8984"}
STYLES = {"response": "-", "t_learner": "--", "x_learner": (0, (5, 1, 1, 1)), "trans_out": "-.",
          "dr_learner": (0, (3, 1, 1, 1, 1, 1)), "causal_forest": "-", "hybrid": (0, (6, 2)), "random": (0, (1, 2))}
MARKERS = {"response": "o", "t_learner": "s", "x_learner": "v", "trans_out": "^", "dr_learner": "P",
           "causal_forest": "D", "hybrid": "*"}
WIDTH = {"response": 1.6, "causal_forest": 1.6, "hybrid": 1.4}
COL_W = 3.45  # inches, IEEE single column
FIG_METHODS = list(ALL_METHODS)  # methods drawn in figures (set in make_all)

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.3,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                     "grid.linewidth": 0.5, "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
                     "font.family": "serif", "pdf.fonttype": 42, "lines.linewidth": 1.1})


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def _line(ax, x, y, m, label=None, **kw):
    ax.plot(x, y, color=COLORS[m], ls=STYLES[m], lw=WIDTH.get(m, 1.1), label=label or SHORT.get(m, m), **kw)


def fig_qini(curves, outcome, out_dir):
    c = curves[curves.outcome == outcome]
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    for m in FIG_METHODS:
        d = c[c.method == m]
        _line(ax, d.phi * 100, d.qini_per_user * 1000, m)
    d = c[c.method == "response"]
    _line(ax, d.phi * 100, d.random_qini_per_user * 1000, "random", label="Random targeting")
    ax.set_xlabel("Users targeted, ranked by model score (%)")
    ax.set_ylabel(f"Incremental {outcome}s per 1,000 users\n(Qini, $Q(k)/N\\times 1000$)")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    _save(fig, out_dir, f"qini_curves_{outcome}")


def fig_topk(main, outcome, out_dir, top_k):
    d = main[main.outcome == outcome]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    n = len(FIG_METHODS)
    width = 0.8 / n
    xs = np.arange(len(top_k))
    for j, m in enumerate(FIG_METHODS):
        rr = [d[(d.method == m) & (d.metric == f"uplift_top{int(round(k * 100))}")].iloc[0] for k in top_k]
        est = np.array([r.estimate for r in rr]) * 100
        lo = np.array([r.ci_low for r in rr]) * 100
        hi = np.array([r.ci_high for r in rr]) * 100
        ax.bar(xs + (j - (n - 1) / 2) * width, est, width * 0.88, color=COLORS[m], label=SHORT[m],
               yerr=[est - lo, hi - est], error_kw={"elinewidth": 0.6, "capsize": 1.2, "ecolor": "#3a3936"})
    ate = d[(d.method == "response") & (d.metric == "ate")].iloc[0].estimate * 100
    ax.axhline(ate, color=COLORS["random"], ls="--", lw=0.9, label=f"All users (ATE {ate:.2f} pp)")
    ax.set_xticks(xs, [f"Top {int(round(k * 100))}%" for k in top_k])
    ax.set_ylabel(f"Observed {outcome} uplift (pp)")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.30), columnspacing=0.8,
              handlelength=1.2)
    ax.grid(axis="x", visible=False)
    _save(fig, out_dir, f"uplift_topk_{outcome}")


def fig_profit(curves, budget, V, c, out_dir):
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    for m in FIG_METHODS + ["random"]:
        d = curves[curves.method == m]
        _line(ax, d.phi * 100, d.profit_test, m, label="Random targeting" if m == "random" else None)
        if m in MARKERS:
            r = budget[budget.policy == m].iloc[0]
            ax.plot(r.phi_selected * 100, r.profit_test, marker=MARKERS[m], color=COLORS[m], ms=5,
                    mec="white", mew=0.7, ls="none")
    ax.axhline(0, color="#52514e", lw=0.6)
    ax.set_xlabel("Targeting fraction $\\phi$ (%)")
    ax.set_ylabel("Test profit per 1,000 users")
    ax.set_xlim(0, 100)
    ax.set_title(f"$V={V:g}$, $c={c:g}$ (illustrative); markers = $\\phi^*$ chosen on validation", fontsize=6.8)
    ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=6)
    _save(fig, out_dir, "profit_vs_targeting_fraction")


def fig_exposure(dec, out_dir, methods=("response", "x_learner", "causal_forest")):
    """Two panels sharing the decile axis: exposure rate (treated) and ITT."""
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.75), sharex=True)
    for m in methods:
        d = dec[dec.method == m]
        axes[0].plot(d.decile, d.exposure_rate_treated * 100, color=COLORS[m], ls=STYLES[m], marker=MARKERS[m],
                     ms=3.2, lw=WIDTH.get(m, 1.1), label=SHORT[m])
        axes[1].plot(d.decile, d.itt * 100, color=COLORS[m], ls=STYLES[m], marker=MARKERS[m], ms=3.2,
                     lw=WIDTH.get(m, 1.1), label=SHORT[m])
    axes[0].set_ylabel("Exposed, treated users (%)")
    axes[1].set_ylabel("Visit uplift, ITT (pp)")
    axes[1].axhline(0, color="#52514e", lw=0.6)
    for ax in axes:
        ax.set_xlabel("Score decile (1 = top)")
        ax.set_xticks(range(1, 11))
        ax.tick_params(axis="x", labelsize=6)
    axes[0].legend(frameon=False, loc="upper right", fontsize=6)
    fig.tight_layout(w_pad=1.0)
    _save(fig, out_dir, "exposure_by_decile")


# ----------------------------------------------------------------------------- tables
def _num(x, nd, sign=False):
    """Format a number for LaTeX: no negative zero, proper minus sign."""
    v = round(float(x), nd) + 0.0
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", "$-$")


def _fmt_ci(est, lo, hi, scale, nd, sign=False):
    return f"{_num(est * scale, nd, sign)} [{_num(lo * scale, nd)}, {_num(hi * scale, nd)}]"


def _write(out_dir, name, body):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name + ".tex"), "w") as fh:
        fh.write("% Auto-generated by src/report.py from results/*.csv -- do not edit by hand.\n" + body)


def tab_profile(prof, quality, out_dir):
    p = prof.set_index("metric")
    q = quality.set_index("metric").value

    def row(label, key, scale=1.0, nd=2, intfmt=False):
        f, s = p.loc[key, "full_file"], p.loc[key, "sample"]
        if intfmt:
            return f"{label} & {int(f):,} & {int(s):,} \\\\\n"
        return f"{label} & {f * scale:.{nd}f} & {s * scale:.{nd}f} \\\\\n"

    body = ("\\begin{tabular}{lrr}\n\\toprule\n & Full file & Sample \\\\\n\\midrule\n"
            + row("Users", "n_rows", intfmt=True)
            + row("Treated share (\\%)", "treatment_share", 100, 2)
            + row("Visit rate, treated (\\%)", "visit_rate_treated", 100, 3)
            + row("Visit rate, control (\\%)", "visit_rate_control", 100, 3)
            + row("Visit ATE (pp)", "visit_ate", 100, 3)
            + row("Conversion rate, treated (\\%)", "conversion_rate_treated", 100, 3)
            + row("Conversion rate, control (\\%)", "conversion_rate_control", 100, 3)
            + row("Conversion ATE (pp)", "conversion_ate", 100, 3)
            + row("Exposure rate, treated (\\%)", "exposure_rate_treated", 100, 2)
            + f"Exact duplicate rows & {int(q['exact_duplicate_rows_full_file']):,} & \\\\\n"
            + "\\midrule\n"
            + "Train / valid / test & \\multicolumn{2}{r}{" + " / ".join(
                f"{int(p.loc[f'n_{s}', 'sample']):,}" for s in ["train", "valid", "test"]) + "} \\\\\n"
            + "\\bottomrule\n\\end{tabular}\n")
    _write(out_dir, "tab_dataset_profile", body)


def tab_randomization(rc, imb, out_dir):
    lines = "".join(
        f"{r.feature} & {_num(r.mean_treated, 3)} & {_num(r.mean_control, 3)} & {_num(r.smd, 4, True)} & "
        f"{_num(r.smd_full_file, 4, True)} & {_num(r.z_full_file, 1, True)} \\\\\n" for r in rc.itertuples())
    auc = imb.set_index("analysis").loc["treatment_classifier_auc_test"]
    sd = rc.sd_smd_under_randomization_full_file.iloc[0]
    body = ("\\begin{tabular}{lrrrrr}\n\\toprule\nFeature & Mean (T) & Mean (C) & SMD & SMD (full) & $z$ (full) \\\\\n"
            "\\midrule\n" + lines + "\\midrule\n"
            + f"\\multicolumn{{6}}{{l}}{{SD of SMD expected under randomization (full file): {sd:.4f}}} \\\\\n"
            + f"\\multicolumn{{6}}{{l}}{{Classifier AUC, treatment from features (test): "
              f"{auc.value:.4f} [{auc.ci_low:.4f}, {auc.ci_high:.4f}]}} \\\\\n"
            "\\bottomrule\n\\end{tabular}\n")
    _write(out_dir, "tab_randomization", body)


def tab_main(main, diffs, out_dir, top_k, outcome="visit"):
    ks = [f"uplift_top{int(round(k * 100))}" for k in top_k]
    d = main[main.outcome == outcome]
    dd = diffs[diffs.outcome == outcome]
    ate = d[(d.method == "response") & (d.metric == "ate")].iloc[0]
    body = ("\\begin{tabular}{l" + "c" * (1 + len(ks)) + "}\n\\toprule\nMethod & AUQC ($\\times10^3$) & "
            + " & ".join(f"Top {k[10:]}\\% uplift (pp)" for k in ks) + " \\\\\n\\midrule\n"
            + f"\\multicolumn{{{2 + len(ks)}}}{{l}}{{\\textit{{Estimates (test ATE {ate.estimate * 100:.3f} pp)}}}} \\\\\n")
    metrics = [("auqc", 1000, 3)] + [(k, 100, 2) for k in ks]
    for m in ALL_METHODS:
        cells = []
        for k, sc, nd in metrics:
            r = d[(d.method == m) & (d.metric == k)].iloc[0]
            cells.append(_fmt_ci(r.estimate, r.ci_low, r.ci_high, sc, nd))
        body += f"{SHORT[m]} & " + " & ".join(cells) + " \\\\\n"
    body += (f"\\midrule\n\\multicolumn{{{2 + len(ks)}}}{{l}}{{\\textit{{Paired difference to the response model"
             f" ($^{{*}}$: significant after Holm correction over all {len(dd)} comparisons)}}}} \\\\\n")
    for m in ALL_METHODS[1:]:
        cells = []
        for k, sc, nd in metrics:
            r = dd[(dd.method == m) & (dd.metric == k)].iloc[0]
            star = "$^{*}$" if r.significant_holm_5pct else ""
            cells.append(_fmt_ci(r.difference_vs_response, r.ci_low, r.ci_high, sc, nd, sign=True) + star)
        body += f"{SHORT[m]} & " + " & ".join(cells) + " \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    _write(out_dir, "tab_main_results", body)


def tab_conversion(main, diffs, out_dir):
    d = main[main.outcome == "conversion"]
    dd = diffs[(diffs.outcome == "conversion") & (diffs.metric == "auqc")]
    lines = ""
    for m in ALL_METHODS:
        r = d[(d.method == m) & (d.metric == "auqc")].iloc[0]
        diff = "--"
        if m != "response":
            q = dd[dd.method == m].iloc[0]
            diff = _fmt_ci(q.difference_vs_response, q.ci_low, q.ci_high, 1000, 3, sign=True)
        lines += f"{SHORT[m]} & {_fmt_ci(r.estimate, r.ci_low, r.ci_high, 1000, 3)} & {diff} \\\\\n"
    body = ("\\begin{tabular}{lll}\n\\toprule\nMethod & AUQC ($\\times10^3$) & $\\Delta$ vs Resp. \\\\\n\\midrule\n"
            + lines + "\\bottomrule\n\\end{tabular}\n")
    _write(out_dir, "tab_conversion", body)


def tab_budget(budget, out_dir):
    names = {"contact_none": "Contact nobody", "contact_all": "Contact everyone",
             "random": "Random targeting", **SHORT}

    def ci(lo, hi):
        return "" if pd.isna(lo) else f" [{_num(lo, 1)}, {_num(hi, 1)}]"

    lines = ""
    for r in budget.itertuples():
        diff = "--" if pd.isna(getattr(r, "diff_vs_response", np.nan)) else \
            _num(r.diff_vs_response, 1, True) + ci(r.diff_ci_low, r.diff_ci_high)
        orc = "--" if pd.isna(r.profit_test_oracle) else \
            f"{_num(r.profit_test_oracle, 1)} ({r.phi_test_oracle * 100:.0f}\\%)"
        lines += (f"{names[r.policy]} & {r.phi_selected * 100:.0f}\\% & {_num(r.profit_test, 1)}"
                  f"{ci(r.profit_test_ci_low, r.profit_test_ci_high)} & {diff} & {orc} \\\\\n")
    body = ("\\begin{tabular}{lrlll}\n\\toprule\nPolicy & $\\phi^*$ & Test profit [95\\% CI] & "
            "$\\Delta$ vs Resp. [95\\% CI] & Oracle ($\\phi$) \\\\\n\\midrule\n" + lines
            + "\\bottomrule\n\\end{tabular}\n")
    _write(out_dir, "tab_budget", body)


def tab_seeds(summ, out_dir):
    s = summ.set_index("method")
    n = int(s.n_seeds.max())
    lines = ""
    for m in ALL_METHODS:
        r = s.loc[m]
        win = "--" if m == "response" else f"{int(r.auqc_seeds_better_than_response)}/{n}"
        winp = "--" if m == "response" else f"{int(r.profit_test_seeds_better_than_response)}/{n}"
        lines += (f"{SHORT[m]} & {r.auqc_mean * 1000:.2f} $\\pm$ {r.auqc_sd * 1000:.2f} & {win} & "
                  f"{r.uplift_top10_mean * 100:.2f} $\\pm$ {r.uplift_top10_sd * 100:.2f} & "
                  f"{r.profit_test_mean:.0f} $\\pm$ {r.profit_test_sd:.0f} & {winp} \\\\\n")
    body = ("\\begin{tabular}{lccccc}\n\\toprule\n & \\multicolumn{2}{c}{AUQC ($\\times10^3$)} & Top 10\\% (pp) & "
            "\\multicolumn{2}{c}{Profit per 1,000} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}\n"
            "Method & mean $\\pm$ SD & $>$Resp. & mean $\\pm$ SD & mean $\\pm$ SD & $>$Resp. \\\\\n\\midrule\n"
            + lines + "\\bottomrule\n\\end{tabular}\n")
    _write(out_dir, "tab_seeds", body)


def tab_exposure(expo, out_dir):
    lines = ""
    for r in expo.itertuples():
        lines += (f"{r.outcome.capitalize()} & {_fmt_ci(r.itt, r.itt_ci_low, r.itt_ci_high, 100, 3)} & "
                  f"{r.exposure_rate_treated * 100:.2f} & {_fmt_ci(r.cace, r.cace_ci_low, r.cace_ci_high, 100, 2)} \\\\\n")
    body = ("\\begin{tabular}{llcl}\n\\toprule\nOutcome & ITT (pp) [95\\% CI] & Exposed (\\%) & "
            "CACE (pp) [95\\% CI] \\\\\n\\midrule\n" + lines + "\\bottomrule\n\\end{tabular}\n")
    _write(out_dir, "tab_exposure", body)


def _plot_methods(results):
    """Methods drawn in figures: the hybrid is left out when its validation-chosen
    weight is 0 or 1, because it is then identical to one of its components."""
    f = os.path.join(results, "tuning.csv")
    if os.path.exists(f):
        t = pd.read_csv(f)
        h = t[(t.method == "hybrid") & (t.outcome == "visit") & (t.selected == True)]  # noqa: E712
        if len(h) and json.loads(h.iloc[0].hyperparameters)["weight_response"] in (0.0, 1.0):
            return [m for m in ALL_METHODS if m != "hybrid"]
    return list(ALL_METHODS)


def make_all(cfg, results):
    global FIG_METHODS
    FIG_METHODS = _plot_methods(results)
    fig_dir, tab_dir = os.path.join(results, "figures"), os.path.join(results, "tables")
    rd = lambda f: pd.read_csv(os.path.join(results, f))  # noqa: E731
    top_k = cfg["evaluation"]["top_k"]
    curves, main, diffs, budget = rd("qini_curves.csv"), rd("main_results.csv"), rd("paired_differences.csv"), \
        rd("budget_simulation.csv")
    for outcome in [cfg["outcomes"]["primary"], cfg["outcomes"]["secondary"]]:
        fig_qini(curves, outcome, fig_dir)
        fig_topk(main, outcome, fig_dir, top_k)
    fig_profit(rd("profit_curves.csv"), budget, cfg["budget"]["V"], cfg["budget"]["c"], fig_dir)
    tab_profile(rd("dataset_profile.csv"), rd("data_quality.csv"), tab_dir)
    tab_randomization(rd("randomization_check.csv"), rd("imbalance_tests.csv"), tab_dir)
    tab_main(main, diffs, tab_dir, top_k)
    tab_conversion(main, diffs, tab_dir)
    tab_budget(budget, tab_dir)
    tab_exposure(rd("exposure_full.csv"), tab_dir)
    if os.path.exists(os.path.join(results, "exposure_by_decile.csv")):
        fig_exposure(rd("exposure_by_decile.csv"), fig_dir)
    if os.path.exists(os.path.join(results, "seed_summary.csv")):
        tab_seeds(rd("seed_summary.csv"), tab_dir)
