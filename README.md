# Who Should We Market To? Uplift Modeling with Causal ML on the Criteo Uplift Dataset

Course project (Applied Machine Learning for Business, Goa Institute of Management).
Team: Aparna Chaudhary (lead), Pramit Manna, Akash Anant.

## Problem

A standard *response model* ranks customers by how likely they are to act (here: visit
the advertiser's website). Many of those customers would have acted **without** the ad.
*Uplift modeling* instead ranks customers by the estimated **incremental** effect of the
campaign, τ(x) = E[Y(1) − Y(0) | X = x]. We compare one response model and three uplift
methods on a large randomized advertising experiment and turn the rankings into a
profit-based targeting decision.

| Method | Idea | Code |
|---|---|---|
| Response model (baseline) | LightGBM P(visit \| X), ignores treatment | `src/models.py` |
| Two-model (T-learner) | separate LightGBM models for treated and control, score = difference | `src/models.py` |
| Transformed outcome | LightGBM regression on Z = Y·T/p − Y·(1−T)/(1−p), p = treated share (≈0.85) | `src/models.py` |
| Causal forest | EconML `CausalForestDML`, trained on a 200,000-row subsample of the training split | `src/models.py` |

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
  train / validation / test, stratified by treatment × visit.
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

Individual steps: `--step prepare | train | evaluate | budget | report`.
Quick test on synthetic data with the same columns (≈30 s, writes elsewhere):

```bash
python -m src.pipeline --step all --synthetic 200000 --results /tmp/synthetic_results --root /tmp/synthetic
```

### Expected runtime

| Step | 4-core Linux machine (our run) | Free Colab (2 cores, estimate) |
|---|---|---|
| download (311 MB) | < 1 min | 1–3 min |
| prepare (read 14M rows, sample, split, checks) | 31 s | 2–4 min |
| train (8 models: 4 methods × 2 outcomes) | ≈ 8 min (causal forest for visit: 3.4 min) | 20–35 min (causal forest < 10 min) |
| evaluate (200 paired bootstrap resamples) | ≈ 1 min | 2–4 min |
| budget + report | < 1 min | 1–2 min |

All seeds are fixed in `config.yaml`; re-running gives identical model scores
(checked: maximum absolute difference 0 for LightGBM, 2·10⁻¹⁷ for the causal forest).

## Configuration

[`config.yaml`](config.yaml) holds every setting: seed (42), sample size (2,000,000),
split ratios (0.6/0.2/0.2), model hyperparameters, causal-forest subsample (200,000),
bootstrap resamples (200), top-k fractions (10/20/30%), and the illustrative budget
parameters (main scenario V = 100 per incremental visit, c = 1.5 per contact;
sensitivity grid V ∈ {50, 100, 200}, c ∈ {0.5, 1, 1.5, 2, 3}).

## Repository layout

```
config.yaml                  all settings
requirements.txt             exact package versions used
src/data.py                  download + checksum, sampling, stratified split, synthetic data
src/models.py                the four methods
src/metrics.py               Qini curve, AUQC, top-k uplift, profit, bootstrap
src/pipeline.py              the steps (prepare/train/evaluate/budget/report)
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
| Table: dataset profile | `results/dataset_profile.csv` → `results/tables/tab_dataset_profile.tex` |
| Table: randomization check (SMD) | `results/randomization_check.csv` → `results/tables/tab_randomization.tex` |
| Table: main results (AUQC, top-k uplift, 95% CIs, paired differences) | `results/main_results.csv`, `results/paired_differences.csv` → `results/tables/tab_main_results.tex` |
| Table: budget simulation | `results/budget_simulation.csv` → `results/tables/tab_budget.tex` |
| Fig.: Qini curves (visit) | `results/figures/qini_curves_visit.{pdf,png}` (data: `results/qini_curves.csv`) |
| Fig.: uplift in top 10/20/30% with CIs | `results/figures/uplift_topk_visit.{pdf,png}` |
| Fig.: profit vs targeting fraction | `results/figures/profit_vs_targeting_fraction.{pdf,png}` (data: `results/profit_curves.csv`) |
| Sensitivity grid over (V, c) | `results/sensitivity_grid.csv` |
| Conversion (secondary) figures | `results/figures/*_conversion.{pdf,png}` |
| Model details (trees, training rows, fit time) | `results/model_info.csv` |
| Rank correlation between model scores (Discussion) | `results/score_correlations.csv` |
| Console log of the real run | `results/run_log.txt` |
| Software versions of the run | `results/run_environment.json` |

**Metric definitions** (implemented in `src/metrics.py`). Users are sorted by model score.
Qini curve Q(k) = Y_T(k) − Y_C(k)·N_T(k)/N_C(k). AUQC = area between the model's Qini
curve and the random-targeting line, with both axes divided by N (reported ×1000, i.e.
per 1,000 users). Top-k uplift = visit rate of treated minus control among the top k% of
users. Profit per 1,000 users Π(φ) = 1000·(V·U(φN)/N − c·φ), where
U(k) = k·(ȳ_T(k) − ȳ_C(k)) are the incremental visits if all top-k users are contacted;
φ is chosen on the validation split and profit is reported on the test split.
Confidence intervals: 200 paired bootstrap resamples of test users (percentile 95% CIs);
the ranking is fixed and the same resampled users are used for every method.

## Citation of the data

E. Diemert, A. Betlei, C. Renaudin, M.-R. Amini. *A Large Scale Benchmark for Uplift
Modeling.* AdKDD & TargetAd Workshop, KDD 2018.

## AI use

Code in this repository was written with the help of an AI assistant (Claude, by
Anthropic) and was reviewed and run by the team. See the AI Use Declaration in the paper.
