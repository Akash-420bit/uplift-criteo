# Data dictionary — Criteo Uplift Prediction Dataset v2.1

Source: Criteo AI Lab, <https://ailab.criteo.com/criteo-uplift-prediction-dataset/>
(Diemert et al., AdKDD 2018). File `criteo-research-uplift-v2.1.csv.gz`,
311,422,618 bytes, SHA-256 `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`.
13,979,592 rows, 16 columns. Licence: CC BY-NC-SA 4.0 (non-commercial use).

The data come from an incrementality test: a random part of the user population
was prevented from being targeted by advertising (control group).

| Column | Type | Meaning | Used in this project as |
|---|---|---|---|
| `f0` … `f11` | float | 12 anonymised user features (dense, projected by Criteo; no business meaning is published) | model inputs `X` |
| `treatment` | 0/1 | **randomly assigned** group: 1 = could be targeted by ads, 0 = held out (control). About 85% are 1. | treatment `T` |
| `visit` | 0/1 | user visited the advertiser's website during the test | **primary outcome** `Y` |
| `conversion` | 0/1 | user converted (bought) during the test; very rare (~0.3%) | secondary outcome |
| `exposure` | 0/1 | user was actually shown an ad (only possible if `treatment` = 1) | **not used** for modelling: it is chosen after randomization, so conditioning on it would break the experiment; only described in the data profile |

## Derived quantities

| Name | Definition |
|---|---|
| treatment share `p` | mean of `treatment` in the training split (≈ 0.85) |
| transformed outcome `Z` | `Y*T/p − Y*(1−T)/(1−p)`; its expectation given `X` equals the uplift τ(X) |
| uplift τ(x) | `E[Y(1) − Y(0) | X = x]`, the effect of being targetable on the outcome |
| split | `train` 60% / `valid` 20% / `test` 20%, stratified by `treatment` × `visit`, seed 42 |

## Pipeline files

| File | Content |
|---|---|
| `data/criteo-research-uplift-v2.1.csv.gz` | raw download (not in git) |
| `data/sample_split.npz` | the 2,000,000-row simple random sample with its split labels (not in git) |
| `results/predictions/*.npz` | model scores on validation and test (not in git, ~60 MB) |
| `results/*.csv`, `results/figures/`, `results/tables/` | all reported numbers, figures and LaTeX tables (in git) |
