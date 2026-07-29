#!/usr/bin/env python3
"""Round-2 shared statistics helpers.

Loaders and small statistical utilities shared by the round-2 re-analysis
scripts (paired_ratios.py, variance_components.py, scaling_sensitivity.py,
scaling_decomp.py, parallel_speedup.py). These operate on the RAW per-repetition
timing CSVs already produced by the benchmark harness -- NO new benchmark runs.

The experimental hierarchy is respected explicitly:

    repetition  subset of  RHS (seed)  subset of  (matrix, n)  subset of  family

so a "unit" here is one (language, regime, phase, family, instance_key, n, nnz,
seed): the finest cell that owns a bag of technical repetitions. Julia and Python
share every (regime, phase, family, instance_key, n, nnz, seed) cell by
construction, which is what makes the cross-language comparison PAIRED.

stdlib csv + numpy only (scipy only where a caller needs it).
"""
from __future__ import annotations

import csv
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.normpath(os.path.join(HERE, ".."))
RESULTS_DIR = os.path.join(CODE_DIR, "results")
BENCH_DIR = os.path.join(RESULTS_DIR, "benchmarks")

TIMING_FILES = (("bench_spd_timing.csv", "spd"),
                ("bench_drazin_timing.csv", "drazin"))

# The cell that owns a bag of technical repetitions. Language is kept SEPARATE
# from this key so the same cell can be paired across the two languages.
UNIT_KEYS = ("phase", "regime", "method", "family", "instance_key", "n", "nnz",
             "seed")

# size-encoding tokens co-vary with n; every other instance-label token is a
# held-fixed secondary parameter. Identical to aggregate.family_group so the
# round-2 fits line up with the round-1 per-family ladders.
_SIZE_TOKEN = re.compile(r"^(n\d+|m\d+|ncore\d+)$")


def family_group(family, instance_key):
    base = instance_key.split("/")[-1]
    kept = [t for t in base.split("_") if not _SIZE_TOKEN.match(t)]
    label = "_".join(kept)
    return f"{family}/{label}" if label else family


def _f(x):
    if x is None or x == "" or x == "NaN":
        return float("nan")
    try:
        return float(x)
    except ValueError:
        return float("nan")


def load_timing_rows(bench_dir=BENCH_DIR, files=TIMING_FILES):
    """Every raw per-repetition timing row, phase-tagged."""
    rows = []
    for fn, phase in files:
        path = os.path.join(bench_dir, fn)
        if not os.path.isfile(path):
            continue
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                r["phase"] = phase
                rows.append(r)
    return rows


def unit_samples(rows, value_col="t_solve_ns"):
    """Group raw rows into units and return, per (unit_key + language):

        {"key": dict of UNIT_KEYS, "language": str, "family_group": str,
         "samples": np.ndarray of the per-rep value_col (NaNs dropped),
         "median": float}

    One entry per (unit, language). The technical repetitions live in
    "samples"; a single robust number is the per-unit median.
    """
    groups = {}
    for r in rows:
        key = tuple(r[k] for k in UNIT_KEYS) + (r["language"],)
        groups.setdefault(key, []).append(r)
    out = []
    for key, recs in groups.items():
        d = dict(zip(UNIT_KEYS, key[:-1]))
        lang = key[-1]
        s = np.array([_f(r[value_col]) for r in recs], dtype=float)
        s = s[np.isfinite(s)]
        out.append({
            "key": d,
            "language": lang,
            "family_group": family_group(d["family"], d["instance_key"]),
            "samples": s,
            "median": float(np.median(s)) if s.size else float("nan"),
            "n": int(d["n"]),
            "nnz": int(d["nnz"]),
            "n_reps": int(s.size),
        })
    return out


def pair_by_cell(units):
    """Collapse a list of unit dicts (from unit_samples) into paired cells keyed
    by the language-independent UNIT_KEYS. Returns
        {cell_key_tuple: {"julia": unit, "python": unit, ...meta}}
    keeping only cells present in BOTH languages.
    """
    cells = {}
    for u in units:
        ck = tuple(u["key"][k] for k in UNIT_KEYS)
        cells.setdefault(ck, {})[u["language"]] = u
    paired = {}
    for ck, langs in cells.items():
        if "julia" in langs and "python" in langs:
            paired[ck] = langs
    return paired


def bootstrap_ci(values, stat=np.median, B=5000, alpha=0.05, seed=20260728):
    """Nonparametric bootstrap CI of `stat` over a 1-D sample (resamples the
    OBSERVATION units, i.e. the array rows the caller passes in)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(stat(v))
    if v.size == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(B, v.size))
    boots = stat(v[idx], axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def ols_loglog(n_vals, T_vals):
    """OLS of log T = b0 + b1 log n over the points AS GIVEN (the caller decides
    the observation level). Returns a dict with beta0/beta1, the analytic t-based
    95% CI on beta1 with df = (#points - 2), stderr, R^2, df, and #points."""
    n = np.asarray(n_vals, dtype=float)
    T = np.asarray(T_vals, dtype=float)
    good = (n > 0) & np.isfinite(T) & (T > 0)
    n, T = n[good], T[good]
    res = {"n_points": int(n.size), "distinct_n": int(np.unique(n).size)}
    nan = float("nan")
    if n.size < 3:
        res.update(beta0=nan, beta1=nan, beta1_ci_lo=nan, beta1_ci_hi=nan,
                   beta1_stderr=nan, r_squared=nan, df=max(n.size - 2, 0))
        return res
    from scipy import stats
    x, y = np.log(n), np.log(T)
    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    df = x.size - 2
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else nan
    if df > 0:
        s2 = ss_res / df
        se1 = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1]))
        tcrit = float(stats.t.ppf(0.975, df))
        lo, hi = beta[1] - tcrit * se1, beta[1] + tcrit * se1
    else:
        se1 = lo = hi = nan
    res.update(beta0=float(beta[0]), beta1=float(beta[1]),
               beta1_ci_lo=float(lo), beta1_ci_hi=float(hi),
               beta1_stderr=float(se1), r_squared=float(r2), df=int(df))
    return res


def per_n_median(units_or_pairs, value="median", lang=None):
    """Collapse a bag of unit dicts to ONE value per distinct n: the median over
    the RHS-level unit values at that n. This is the CORRECT observation level
    for a scaling fit (one independent point per problem size, RHS medians
    pooled). Returns sorted (n_array, T_array)."""
    by_n = {}
    for u in units_or_pairs:
        if lang is not None and u["language"] != lang:
            continue
        by_n.setdefault(u["n"], []).append(u[value])
    ns = sorted(by_n)
    T = [float(np.median(by_n[n])) for n in ns]
    return np.array(ns, dtype=float), np.array(T, dtype=float)


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")
