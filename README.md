# Who Should We Market To? Uplift Modeling with Causal ML on the Criteo Uplift Dataset

Course project (Applied Machine Learning for Business, Goa Institute of Management).
Team: Aparna Chaudhary (lead), Pramit Manna, Akash Anant.

## Problem

A standard *response model* ranks customers by how likely they are to act (here: visit
the advertiser's website). Many of those customers would have acted **without** the ad.
*Uplift modeling* instead ranks customers by the estimated **incremental** effect of the
campaign, τ(x) = E[Y(1) − Y(0) | X = x]. We compare one response model, five uplift
methods and a hybrid score on a large randomized advertising experiment and turn the rankings into a
profit-based targeting decision.

| Method | Idea |
|---|---|
| Response model (baseline) | LightGBM P(visit \| X), ignores treatment |
| T-learner | separate LightGBM models for treated and control; score = difference |
| X-learner | imputed individual effects, combined with the treated share (Künzel et al. 2019) |
| Transformed outcome | LightGBM regression on Z = Y·T/p − Y·(1−T)/(1−p), p = treated share (≈0.85) |
| DR-learner | regression on the doubly robust pseudo-outcome, cross-fitted nuisance models (Kennedy 2023) |
| Causal forest | EconML `CausalForestDML` (a forest R-learner), all training users |
| Hybrid | rank blend of the response score and the best uplift score; weight chosen on validation |

Every method is trained on the same 1.2M training users and its hyperparameters are
chosen by the **Qini coefficient on the validation split** (grids in `config.yaml`).
Code: `src/models.py`.

**Extra analyses** (all from the same run):
* **Exposure / complier effect** – using `treatment` as an instrument for `exposure`,
  the effect of the ad on users who actually saw it (CACE), overall and by score decile.
* **Randomization audit** – z-tests of feature differences, a classifier that tries to
  predict treatment from the features, and regression-adjusted vs. unadjusted effects.
* **Data quality** – 11.8% of the published rows are exact duplicates; the split keeps
  identical feature vectors together so no test user also appears in training.
* **Robustness** – the whole pipeline (new 2M sample, split, tuning) is repeated for
  5 seeds; test comparisons use 1,000 paired bootstrap resamples with Holm correction.

## Data

Criteo Uplift Prediction Dataset v2.1 (Diemert et al., 2018),
<https://ailab.criteo.com/criteo-uplift-prediction-dataset/>, licence CC BY-NC-SA 4.0.
13,979,592 users, 12 anonymised features `f0`–`f11`, randomized `treatment` (85% treated),
outcomes `visit` and `conversion`, and `exposure`. See [`docs/data_dictionary.md`](docs/data_dictionary.md).

* The pipeline downloads the file automatically and **verifies its SHA-256 checksum**
  against the checksum of the official Criteo file. (The official link
  `go.criteo.net/...` redirects to a Criteo storage server that was unreachable in
  November 2026, so the default source is a byte-identical GitHub copy; the checksum
  proves it is the same file.)
* We use a **simple random sample of 2,000,000 users** (seed 42), split 60/20/20 into
  train / validation / test, stratified by treatment × visit, with identical feature
  vectors kept in the same split (the file contains many duplicate rows).
* We use the randomized `treatment` column, **not** `exposure`.
* Primary outcome: `visit`. Secondary: `conversion` (very rare, ≈0.3%).
* The raw data is **not** stored in this repository.

## How to reproduce

### Option A — Google Colab (no installation, recommended)

1. Click [**Open the notebook in Colab**](https://colab.research.google.com/github/Akash-420bit/uplift-criteo/blob/main/notebooks/reproduce_colab.ipynb)
   (it opens [`notebooks/reproduce_colab.ipynb`](notebooks/reproduce_colab.ipynb)).
2. Run the cells one by one, top to bottom (▶ button). Each cell says what it does.
3. Everything is saved in your Google Drive under `MyDrive/uplift-criteo-run/`.
   If Colab disconnects, run the cells again from the top; finished work is skipped.
4. The last cell creates `results.zip`.

### Option B — your own computer (Python 3.11–3.13, ~6 GB free RAM)

```bash
git clone https://github.com/Akash-420bit/uplift-criteo
cd uplift-criteo
pip install -r requirements.txt
python -m src.pipeline --step all        # downloads data, runs everything, writes results/
```

Individual steps: `--step prepare | train | evaluate | exposure | budget | seeds | report`.
The `seeds` step (4 extra seeds) is the slowest; everything else does not need it.
Quick test on synthetic data with the same columns (≈30 s, writes elsewhere):

```bash
python -m src.pipeline --step all --synthetic 200000 --results /tmp/synthetic_results --root /tmp/synthetic
```

### Expected runtime

| Step | 4-core Linux machine (our run) | Free Colab (2 cores, estimate) |
|---|---|---|
| download (311 MB) | < 1 min | 1–3 min |
| prepare (read 14M rows, audit, exposure analysis, sample, split) | 1.7 min | 4–6 min |
| train (6 methods × 2–4 settings × 2 outcomes, + hybrid) | ≈ 20 min (causal forest ≈ 80 s per setting) | 45–60 min |
| evaluate (1,000 paired bootstrap resamples, 7 scores × 2 outcomes) | ≈ 6 min | 10–15 min |
| exposure, budget, report | ≈ 2 min | 3–5 min |
| seeds (4 extra seeds, visit only; optional) | ≈ 45 min | 1.5–2.5 h |

All seeds are fixed in `config.yaml`; re-running gives identical model scores
(checked: maximum absolute difference 0 for the X-learner, 8·10⁻¹⁷ for the causal forest).
Do not run two pipelines at once on the same machine: LightGBM slows down sharply when
several processes use all cores.

## Configuration

[`config.yaml`](config.yaml) holds every setting: seed (42), sample size (2,000,000),
split ratios (0.6/0.2/0.2), tuning grids (chosen by validation Qini), causal-forest settings (all training users, 7% per tree),
bootstrap resamples (1,000), extra seeds (1–4), top-k fractions (10/20/30%), and the illustrative budget
parameters (main scenario V = 100 per incremental visit, c = 1.5 per contact;
sensitivity grid V ∈ {50, 100, 200}, c ∈ {0.5, 1, 1.5, 2, 3}).

## Repository layout

```
config.yaml                  all settings
requirements.txt             exact package versions used
src/data.py                  download + checksum, sampling, stratified split, synthetic data
src/models.py                the six methods (hybrid is built in pipeline.py)
src/metrics.py               Qini curve, AUQC, top-k uplift, profit, bootstrap
src/pipeline.py              the steps (prepare/train/evaluate/exposure/budget/seeds/report)
src/report.py                figures (PDF+PNG) and LaTeX tables from the CSVs
notebooks/reproduce_colab.ipynb   one-cell-per-step Colab notebook
docs/data_dictionary.md      column definitions
results/                     all outputs of the real run (see below)
paper/                       IEEE-format LaTeX source of the paper
```

## Results and how they map to the paper

All numbers in the paper come from these files (test split, 400,000 users, unless noted).

| Paper item | File(s) |
|---|---|
| Table: dataset profile (+ duplicate rows) | `results/dataset_profile.csv`, `results/data_quality.csv` → `results/tables/tab_dataset_profile.tex` |
| Table: randomization audit (SMD, z-tests, classifier AUC) | `results/randomization_check.csv`, `results/imbalance_tests.csv` → `results/tables/tab_randomization.tex` |
| Regression-adjusted vs unadjusted effect | `results/imbalance_tests.csv` |
| Table: main results, visit (AUQC, top-k, CIs, paired differences, Holm) | `results/main_results.csv`, `results/paired_differences.csv` → `results/tables/tab_main_results.tex` |
| Conversion results | same files (outcome = conversion) → `results/tables/tab_conversion.tex` |
| Table: robustness over 5 seeds | `results/seed_results.csv`, `results/seed_summary.csv` → `results/tables/tab_seeds.tex` |
| Table: ITT and complier effect (CACE) | `results/exposure_full.csv` → `results/tables/tab_exposure.tex` |
| Fig.: exposure and uplift by score decile | `results/exposure_by_decile.csv`, `results/exposure_summary.csv` → `results/figures/exposure_by_decile.{pdf,png}` |
| Table: budget simulation | `results/budget_simulation.csv` → `results/tables/tab_budget.tex` |
| Fig.: Qini curves (visit) | `results/figures/qini_curves_visit.{pdf,png}` (data: `results/qini_curves.csv`) |
| Fig.: uplift in top 10/20/30% with CIs | `results/figures/uplift_topk_visit.{pdf,png}` |
| Fig.: profit vs targeting fraction | `results/figures/profit_vs_targeting_fraction.{pdf,png}` (data: `results/profit_curves.csv`) |
| Sensitivity grid over (V, c) | `results/sensitivity_grid.csv` |
| Tuning (all settings, validation AUQC used for selection; test AUQC recorded for reporting only) and hybrid weights | `results/tuning.csv`, `results/seed_tuning.csv` |
| Rank correlation between model scores | `results/score_correlations.csv` |
| Software versions / console log of the run | `results/run_environment.json`, `results/run_log.txt` |

**Metric definitions** (implemented in `src/metrics.py`). Users are sorted by model score.
Qini curve Q(k) = Y_T(k) − Y_C(k)·N_T(k)/N_C(k). AUQC = area between the model's Qini
curve and the random-targeting line, with both axes divided by N (reported ×1000, i.e.
per 1,000 users). Top-k uplift = visit rate of treated minus control among the top k% of
users. Profit per 1,000 users Π(φ) = 1000·(V·U(φN)/N − c·φ), where
U(k) = k·(ȳ_T(k) − ȳ_C(k)) are the incremental visits if all top-k users are contacted;
φ is chosen on the validation split and profit is reported on the test split.
Confidence intervals: 1,000 paired bootstrap resamples of test users (percentile 95% CIs);
the ranking is fixed and the same resampled users are used for every method. Two-sided
bootstrap p-values of the 24 comparisons with the response model (per outcome) are
Holm-corrected. Complier effect: CACE = ITT / P(exposed | treated), delta-method CI.

## Citation of the data

E. Diemert, A. Betlei, C. Renaudin, M.-R. Amini. *A Large Scale Benchmark for Uplift
Modeling.* AdKDD & TargetAd Workshop, KDD 2018.

## AI use

Code in this repository was written with the help of an AI assistant (Claude, by
Anthropic) and was reviewed and run by the team. See the AI Use Declaration in the paper.
