"""Data loading, sampling, splitting and a synthetic generator for testing."""
from __future__ import annotations

import hashlib
import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [f"f{i}" for i in range(12)]
COLUMNS = FEATURES + ["treatment", "conversion", "visit", "exposure"]
DTYPES = {**{f: "float32" for f in FEATURES},
          "treatment": "int8", "conversion": "int8", "visit": "int8", "exposure": "int8"}


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(cfg: dict, root: str) -> str:
    """Download the raw file if missing and verify its SHA-256 checksum."""
    path = os.path.join(root, cfg["data"]["raw_path"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        last_err = None
        for url in (cfg["data"]["url_mirror"], cfg["data"]["url_official"]):
            try:
                print(f"Downloading {url} ...")
                urllib.request.urlretrieve(url, path + ".part")
                os.replace(path + ".part", path)
                break
            except Exception as err:  # try the next source
                last_err = err
                print(f"  failed: {err}")
        else:
            raise RuntimeError(f"Could not download the data: {last_err}")
    digest = sha256_of(path)
    if digest != cfg["data"]["sha256"]:
        raise RuntimeError(f"Checksum mismatch for {path}: {digest}")
    print(f"Checksum OK ({digest[:12]}...)")
    return path


def load_full(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=DTYPES, usecols=COLUMNS)[COLUMNS]


def simple_random_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, replace=False, random_state=seed).reset_index(drop=True)


def split(df: pd.DataFrame, cfg: dict, seed: int) -> pd.Series:
    """Return a Series with values train/valid/test.

    About 12% of the rows in the published file are exact duplicates, so users
    with an identical feature vector are kept in the SAME split (grouped split);
    otherwise a model could be evaluated on rows it was trained on. Groups are
    assigned stratified by treatment x visit of their first row.
    """
    s = cfg["split"]
    key = pd.util.hash_pandas_object(df[FEATURES], index=False).to_numpy()
    groups, first, inv = np.unique(key, return_index=True, return_inverse=True)
    strata = (df[cfg["treatment_col"]].to_numpy().astype(int) * 2
              + df[cfg["outcomes"]["primary"]].to_numpy().astype(int))[first]
    g = np.arange(len(groups))
    g_train, g_rest = train_test_split(g, train_size=s["train"], stratify=strata, random_state=seed)
    rel_valid = s["valid"] / (s["valid"] + s["test"])
    g_valid, g_test = train_test_split(g_rest, train_size=rel_valid, stratify=strata[g_rest],
                                       random_state=seed + 1)
    code = np.zeros(len(groups), dtype=np.int8)
    code[g_valid], code[g_test] = 1, 2
    return pd.Series(np.array(["train", "valid", "test"], dtype=object)[code[inv]], index=df.index)


def make_synthetic(n: int, seed: int) -> pd.DataFrame:
    """Synthetic data with the same columns as Criteo v2.1 and a KNOWN uplift.

    Used only to test that the code runs and that the metrics behave sensibly
    (uplift models should beat the response model here by construction).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12)).astype("float32")
    t = (rng.random(n) < 0.85).astype("int8")
    base = 1 / (1 + np.exp(-(-3.2 + 1.2 * X[:, 0] + 0.5 * X[:, 1])))        # baseline visit prob
    tau = 0.03 * (1 / (1 + np.exp(-2.0 * X[:, 2])))                          # uplift depends on f2 only
    p_visit = np.clip(base + t * tau, 0, 1)
    visit = (rng.random(n) < p_visit).astype("int8")
    conversion = (visit * (rng.random(n) < 0.06 + 0.02 * t * (X[:, 2] > 0))).astype("int8")
    exposure = (t * (rng.random(n) < 0.03)).astype("int8")
    df = pd.DataFrame(X, columns=FEATURES)
    df["treatment"], df["conversion"], df["visit"], df["exposure"] = t, conversion, visit, exposure
    return df
