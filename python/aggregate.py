#!/usr/bin/env python3
"""Phase-4 aggregation — PROTOCOL.md sections 4, 5, 10.

Consumes the TWO-PASS raw per-repetition benchmark CSVs written by run_full.py
(code/results/benchmarks/, schema v2):

  * TIMING pass  -> bench_spd_timing.csv, bench_drazin_timing.csv
      30 reps, 3 RHS, regimes single_thread AND multi_thread. Drives all
      wall-clock statistics, normalized metrics, and the scaling fit.
  * CONVERGENCE pass -> bench_spd_conv.csv, bench_drazin_conv.csv
      1 rep, up to 20 RHS, single_thread only. Drives the iteration-count /
      convergence spread across the seeded right-hand sides.

From the TIMING pass, PER EXPERIMENTAL UNIT, robust statistics are recomputed
DIRECTLY from the raw per-rep samples (never from a single wall-clock number):

  * median, IQR (q75 - q25) and 95% bootstrap CI of t_total and t_solve (section 4)
  * normalized metrics (section 5): time per matvec, ns per nonzero,
    peak bytes per nonzero (unit-level medians)
  * deterministic work counts (matvecs, precond applications, orthogonalizations,
    restart cycles, iterations) verified constant across reps

and PER (method, language, phase, regime) an empirical scaling fit (section 10):

      log T = beta0 + beta1 * log n + eps

fitted by ordinary least squares on the unit-level median t_solve, with a
95% confidence interval on the scaling exponent beta1 (single_thread is the
canonical scaling regime).

From the CONVERGENCE pass, PER (language, regime, phase, family, instance,
method) the iteration-count spread across the seeded RHS is summarized
(count, mean, median, min, max, std) — PROTOCOL.md sections 3.3, 5.

Outputs (written next to the CSVs unless --out-dir given):
  * agg_timing.csv         one row per timing experimental unit, all stats
  * agg_conv.csv           one row per convergence instance x method x language
  * scaling_fits.csv       one row per (method, language, phase, regime) fit
  * aggregate_summary.json  row/unit counts, sanity checks, degenerate ladders,
                            IC(0)-breakdown instances, NaN/inf leak guard

No pandas dependency (stdlib csv + numpy + scipy.stats.t only).
A timing experimental unit is the tuple
  (language, regime, phase, family, instance_key, method, n, seed[=RHS seed]).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.normpath(os.path.join(HERE, ".."))
RESULTS_DIR = os.path.join(CODE_DIR, "results", "benchmarks")

# columns that key one experimental unit (reps vary within it)
UNIT_KEYS = ["language", "regime", "phase", "family", "instance_key",
             "method", "n", "nnz", "seed"]
# deterministic (rep-invariant) integer work counts
WORK_COLS = ["matvecs", "precond_applications", "orthogonalizations",
             "restart_cycles", "iterations", "index_a", "krylov_m"]


def _f(x):
    """Parse a CSV cell to float, mapping NaN/empty to np.nan."""
    if x is None or x == "" or x == "NaN":
        return float("nan")
    return float(x)


def load_rows(path, phase):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            r["phase"] = phase
            rows.append(r)
    return rows


def bootstrap_median_ci(samples, B=2000, alpha=0.05, seed=20260727):
    s = np.asarray(samples, dtype=float)
    s = s[~np.isnan(s)]
    if s.size == 0:
        return float("nan"), float("nan")
    if s.size == 1:
        return float(s[0]), float(s[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, s.size, size=(B, s.size))
    meds = np.median(s[idx], axis=1)
    lo = float(np.percentile(meds, 100 * alpha / 2))
    hi = float(np.percentile(meds, 100 * (1 - alpha / 2)))
    return lo, hi


def nanmedian(samples):
    """np.nanmedian without the all-NaN RuntimeWarning (SPD rows have NaN
    drazin_error by design)."""
    s = np.asarray(samples, dtype=float)
    s = s[~np.isnan(s)]
    return float(np.median(s)) if s.size else float("nan")


def iqr(samples):
    s = np.asarray(samples, dtype=float)
    s = s[~np.isnan(s)]
    if s.size == 0:
        return float("nan")
    return float(np.percentile(s, 75) - np.percentile(s, 25))


def const_or_flag(values):
    """Return the single value if constant across reps, else the median plus a
    'non-constant' note (surfaced so a rough edge is never hidden)."""
    vals = [v for v in values if v is not None]
    uniq = sorted(set(vals))
    if len(uniq) == 1:
        return uniq[0], True
    return int(np.median([int(v) for v in vals])), False


def aggregate_units(rows):
    groups = {}
    for r in rows:
        key = tuple(r[k] for k in UNIT_KEYS)
        groups.setdefault(key, []).append(r)

    units = []
    warnings = []
    for key, recs in sorted(groups.items()):
        d = dict(zip(UNIT_KEYS, key))
        n = int(d["n"])
        nnz = int(d["nnz"])
        t_total = [_f(r["t_total_ns"]) for r in recs]
        t_solve = [_f(r["t_solve_ns"]) for r in recs]
        tpm = [_f(r["time_per_matvec_ns"]) for r in recs]
        nspnz = [_f(r["ns_per_nonzero"]) for r in recs]
        bpnz = [_f(r["peak_bytes_per_nonzero"]) for r in recs]

        u = dict(d)
        u["nnz"] = nnz
        u["n_reps"] = len(recs)
        for name, samp in (("t_total", t_total), ("t_solve", t_solve)):
            lo, hi = bootstrap_median_ci(samp)
            u[f"{name}_median_ns"] = nanmedian(samp)
            u[f"{name}_iqr_ns"] = iqr(samp)
            u[f"{name}_ci_lo_ns"] = lo
            u[f"{name}_ci_hi_ns"] = hi
        u["time_per_matvec_ns_median"] = nanmedian(tpm)
        u["ns_per_nonzero_median"] = nanmedian(nspnz)
        u["peak_bytes_per_nonzero_median"] = nanmedian(bpnz)
        # deterministic work counts (verify constancy across reps)
        for c in WORK_COLS:
            val, ok = const_or_flag([int(r[c]) for r in recs])
            u[c] = val
            if not ok:
                warnings.append(f"{key}: work count {c} not constant across reps")
        # representative diagnostics (median for floats, first for labels)
        u["converged"] = recs[0]["converged"]
        u["stop_residual"] = nanmedian([_f(r["stop_residual"]) for r in recs])
        u["relres"] = nanmedian([_f(r["relres"]) for r in recs])
        u["drazin_error"] = nanmedian([_f(r["drazin_error"]) for r in recs])
        u["precond_kind"] = recs[0]["precond_kind"]
        u["precond_shift"] = _f(recs[0]["precond_shift"])
        u["precond_breakdown"] = recs[0]["precond_breakdown"]
        u["tol"] = _f(recs[0]["tol"])
        u["blas_backend"] = recs[0]["blas_backend"]
        units.append(u)
    return units, warnings


# tokens that encode the problem SIZE (they co-vary with n and must NOT split a
# ladder); everything else (eps, kappa, drazin index k) is a secondary parameter
# that MUST be held fixed within a fit.
_SIZE_TOKEN = re.compile(r"^(n\d+|m\d+|ncore\d+)$")


def family_group(family, instance_key):
    """The n-ladder identity: the matrix family PLUS its held-fixed secondary
    parameter label (eps for anisotropic, kappa for varied-condition, k for the
    singular index), with the size-encoding tokens (n###, m###, ncore###)
    stripped so that only n varies within a group. Mesh/Laplacian families
    reduce to the bare family name (their m###/n### encode only n)."""
    base = instance_key.split("/")[-1]
    kept = [t for t in base.split("_") if not _SIZE_TOKEN.match(t)]
    label = "_".join(kept)
    return f"{family}/{label}" if label else family


def _fit_ladder(us, tvar, base):
    """OLS fit of log T = b0 + b1 log n over one group's units. `base` carries the
    identifying columns. A genuine ladder needs >= 3 distinct n; fewer is reported
    as insufficient_ladder with NaN betas (never a bogus exponent, PROTOCOL 10)."""
    n_arr = np.array([int(u["n"]) for u in us], dtype=float)
    T_arr = np.array([float(u[tvar]) for u in us], dtype=float)
    good = (n_arr > 0) & np.isfinite(T_arr) & (T_arr > 0)
    n_arr, T_arr = n_arr[good], T_arr[good]
    distinct_n = np.unique(n_arr)
    fit = dict(base)
    fit.update(n_points=int(n_arr.size), distinct_n=int(distinct_n.size),
               n_min=int(distinct_n.min()) if distinct_n.size else 0,
               n_max=int(distinct_n.max()) if distinct_n.size else 0,
               t_variable=tvar)
    nan = float("nan")
    if distinct_n.size < 3:
        fit.update(beta0=nan, beta1=nan, beta1_ci_lo=nan, beta1_ci_hi=nan,
                   beta1_stderr=nan, r_squared=nan,
                   status=fit.get("status", "insufficient_ladder"))
        return fit
    x, y = np.log(n_arr), np.log(T_arr)
    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    dof = x.size - 2
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else nan
    if dof > 0:
        s2 = ss_res / dof
        se1 = math.sqrt(float(s2 * np.linalg.inv(X.T @ X)[1, 1]))
        tcrit = stats.t.ppf(0.975, dof)
        ci_lo, ci_hi = beta[1] - tcrit * se1, beta[1] + tcrit * se1
    else:
        se1 = ci_lo = ci_hi = nan
    fit.update(beta0=float(beta[0]), beta1=float(beta[1]),
               beta1_ci_lo=float(ci_lo), beta1_ci_hi=float(ci_hi),
               beta1_stderr=float(se1), r_squared=float(r2),
               status=fit.get("status", "ok"))
    return fit


def scaling_fits(units, tvar="t_solve_median_ns",
                 regimes=("single_thread", "multi_thread")):
    """Empirical scaling fit log T = b0 + b1 log n PER
    (family, method, language, phase, regime) — PROTOCOL.md 10.

    Grouping by the n-ladder identity (family_group) holds every secondary
    parameter (eps, kappa, drazin index) fixed, so each fit is a genuine n-ladder
    rather than a cross-family mixture. Only groups with >= 3 distinct n get an
    exponent; the rest are marked insufficient_ladder. If NO singular family
    reaches 3 distinct n (the dense-n cells were dropped as intractable), one
    explicitly LABELED 'pooled_singular' fallback fit is emitted per
    (method, language, regime) so the singular phase still has a reported
    exponent, with the pooling called out (never silent)."""
    groups = {}
    for u in units:
        reg = u.get("regime")
        if regimes is not None and reg not in regimes:
            continue
        fam = family_group(u["family"], u["instance_key"])
        groups.setdefault((u["phase"], fam, u["language"], u["method"], reg),
                          []).append(u)

    fits = []
    drazin_has_valid = {}  # (lang, method, regime) -> bool
    for (phase, fam, lang, method, regime), us in sorted(groups.items()):
        base = {"phase": phase, "family": fam, "language": lang,
                "method": method, "regime": regime}
        fit = _fit_ladder(us, tvar, base)
        fits.append(fit)
        if phase == "drazin":
            k = (lang, method, regime)
            drazin_has_valid[k] = drazin_has_valid.get(k, False) or (fit["status"] == "ok")

    # Pooled-singular fallback: only where no per-family singular ladder reached 3
    # distinct n. Clearly labeled family='pooled_singular', status='pooled'.
    pooled = {}
    for u in units:
        reg = u.get("regime")
        if u["phase"] != "drazin" or (regimes is not None and reg not in regimes):
            continue
        pooled.setdefault((u["language"], u["method"], reg), []).append(u)
    for (lang, method, regime), us in sorted(pooled.items()):
        if drazin_has_valid.get((lang, method, regime)):
            continue  # a real per-family ladder exists; no pooling needed
        base = {"phase": "drazin", "family": "pooled_singular", "language": lang,
                "method": method, "regime": regime, "status": "pooled"}
        fits.append(_fit_ladder(us, tvar, base))
    return fits


CONV_KEYS = ["language", "regime", "phase", "family", "instance_key",
             "method", "n", "nnz"]


def aggregate_conv(rows):
    """Convergence-pass summary: for each (language, regime, phase, family,
    instance, method) collapse the per-seed iteration counts across the seeded
    RHS into count/mean/median/min/max/std (PROTOCOL.md 3.3, 5). One conv row =
    one RHS seed (the conv pass runs a single rep), so each seed contributes one
    iteration count. relres / drazin_error are summarized as medians."""
    groups = {}
    for r in rows:
        key = tuple(r[k] for k in CONV_KEYS)
        groups.setdefault(key, []).append(r)

    out = []
    for key, recs in sorted(groups.items()):
        d = dict(zip(CONV_KEYS, key))
        iters = np.array([_f(r["iterations"]) for r in recs], dtype=float)
        iters = iters[~np.isnan(iters)]
        matv = np.array([_f(r["matvecs"]) for r in recs], dtype=float)
        matv = matv[~np.isnan(matv)]
        conv_flags = [str(r["converged"]).lower() == "true" for r in recs]
        u = dict(d)
        u["n"] = int(d["n"])
        u["nnz"] = int(d["nnz"])
        u["n_rhs"] = len(recs)
        u["iter_mean"] = float(np.mean(iters)) if iters.size else float("nan")
        u["iter_median"] = float(np.median(iters)) if iters.size else float("nan")
        u["iter_min"] = int(np.min(iters)) if iters.size else -1
        u["iter_max"] = int(np.max(iters)) if iters.size else -1
        u["iter_std"] = float(np.std(iters, ddof=1)) if iters.size > 1 else 0.0
        u["matvec_mean"] = float(np.mean(matv)) if matv.size else float("nan")
        u["converged_frac"] = float(np.mean(conv_flags)) if conv_flags else float("nan")
        u["relres_median"] = nanmedian([_f(r["relres"]) for r in recs])
        u["drazin_error_median"] = nanmedian([_f(r["drazin_error"]) for r in recs])
        u["precond_kind"] = recs[0]["precond_kind"]
        out.append(u)
    return out


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def work_identity_deviation(units):
    """Max |Julia - Python| over shared timing units for the deterministic work
    counts (matvecs, orthogonalizations, precond applications). Equivalent
    implementations must produce byte-identical counts (PROTOCOL.md 2, 5)."""
    by = {}
    for u in units:
        key = (u["phase"], u["method"], u["instance_key"], u["n"], u["seed"],
               u.get("regime"))
        by.setdefault(key, {})[u["language"]] = u
    worst = {c: 0 for c in ("matvecs", "orthogonalizations", "precond_applications")}
    n_shared = 0
    for pair in by.values():
        if "julia" in pair and "python" in pair:
            n_shared += 1
            for c in worst:
                worst[c] = max(worst[c], abs(int(pair["julia"][c]) - int(pair["python"][c])))
    return {"n_shared_units": n_shared, "max_abs_diff": worst}


def drazin_error_contrast(units):
    """Median drazin_error per (instance, method) for the singular phase, so the
    'DGMRES recovers A^D b, GMRES does not' claim is a measured contrast."""
    agg = {}
    for u in units:
        if u["phase"] != "drazin":
            continue
        agg.setdefault((u["instance_key"].split("/")[-1], u["method"]), []).append(
            _f(u["drazin_error"]))
    out = {}
    for (inst, method), vals in sorted(agg.items()):
        out.setdefault(inst, {})[method] = nanmedian(vals)
    return out


def ic0_breakdown_instances(units):
    """SPD instances where pcg_ic0 rows are absent while cg rows are present ->
    the unshifted IC(0) factorization broke down (non-positive pivot) and the cell
    has no timing (PROTOCOL.md 7). Reported as an explicit condition, never a 0/NaN."""
    have = {}
    for u in units:
        if u["phase"] != "spd":
            continue
        have.setdefault(u["instance_key"], set()).add(u["method"])
    broken = sorted(inst for inst, methods in have.items()
                    if "cg" in methods and "pcg_ic0" not in methods)
    return broken


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=RESULTS_DIR)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--spd-timing-csv", default="bench_spd_timing.csv")
    p.add_argument("--drazin-timing-csv", default="bench_drazin_timing.csv")
    p.add_argument("--spd-conv-csv", default="bench_spd_conv.csv")
    p.add_argument("--drazin-conv-csv", default="bench_drazin_conv.csv")
    args = p.parse_args()
    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)
    rd = args.results_dir

    # ---- TIMING pass ----
    timing_rows = (load_rows(os.path.join(rd, args.spd_timing_csv), "spd")
                   + load_rows(os.path.join(rd, args.drazin_timing_csv), "drazin"))
    if not timing_rows:
        raise SystemExit(f"no timing rows under {rd} "
                         f"({args.spd_timing_csv}, {args.drazin_timing_csv})")
    units, warnings = aggregate_units(timing_rows)
    fits = scaling_fits(units)

    # NaN/inf guard on the aggregated timing stats (must never leak into tables)
    leak = []
    for u in units:
        for c in ("t_total_median_ns", "t_solve_median_ns",
                  "t_total_ci_lo_ns", "t_solve_ci_lo_ns"):
            if not math.isfinite(u[c]):
                leak.append(f"{u['language']}/{u['regime']}/{u['method']}/"
                            f"{u['instance_key']}: {c}={u[c]}")

    # ---- CONVERGENCE pass ----
    conv_rows = (load_rows(os.path.join(rd, args.spd_conv_csv), "spd")
                 + load_rows(os.path.join(rd, args.drazin_conv_csv), "drazin"))
    conv = aggregate_conv(conv_rows) if conv_rows else []

    write_csv(os.path.join(out_dir, "agg_timing.csv"), units, list(units[0].keys()))
    write_csv(os.path.join(out_dir, "scaling_fits.csv"), fits, list(fits[0].keys()))
    if conv:
        write_csv(os.path.join(out_dir, "agg_conv.csv"), conv, list(conv[0].keys()))

    # ---- sanity checks (reported, not asserted) ----
    fit_status_counts = {}
    for f in fits:
        fit_status_counts[f["status"]] = fit_status_counts.get(f["status"], 0) + 1
    valid_ladders = sorted({f["family"] for f in fits if f["status"] == "ok"})
    pooled_fits = sorted({f"{f['phase']}/{f['family']}/{f['method']}/{f['language']}/{f['regime']}"
                          for f in fits if f["status"] == "pooled"})
    summary = {
        "timing": {
            "n_raw_rows": len(timing_rows),
            "n_units": len(units),
            "regimes": sorted({u.get("regime", "") for u in units}),
            "phases": sorted({u["phase"] for u in units}),
            "methods": sorted({u["method"] for u in units}),
        },
        "convergence": {
            "n_raw_rows": len(conv_rows),
            "n_units": len(conv),
            "max_rhs_per_unit": max((c["n_rhs"] for c in conv), default=0),
        },
        "n_fits": len(fits),
        "scaling": {
            "grouping": "per (family, method, language, phase, regime)",
            "fit_status_counts": fit_status_counts,
            "valid_ladder_families": valid_ladders,
            "pooled_singular_fits": pooled_fits,
        },
        "work_count_warnings": warnings,
        "timing_nan_leaks": leak,
        "work_identity_check": work_identity_deviation(units),
        "drazin_error_contrast": drazin_error_contrast(units),
        "ic0_breakdown_instances": ic0_breakdown_instances(units),
    }
    with open(os.path.join(out_dir, "aggregate_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"TIMING: {len(timing_rows)} rows -> {len(units)} units, {len(fits)} fits")
    print(f"CONV:   {len(conv_rows)} rows -> {len(conv)} instance-method units")
    print(f"  wrote agg_timing.csv, agg_conv.csv, scaling_fits.csv, "
          f"aggregate_summary.json in {out_dir}")
    wid = summary["work_identity_check"]["max_abs_diff"]
    print(f"  work-count identity max |Jl-Py|: {wid} over "
          f"{summary['work_identity_check']['n_shared_units']} shared units")
    print(f"  IC(0) breakdown instances: {summary['ic0_breakdown_instances']}")
    if warnings:
        print(f"  WARNING: {len(warnings)} non-constant work-count group(s)")
    print("  timing NaN/inf guard: " + ("clean" if not leak else f"{len(leak)} leak(s)"))


if __name__ == "__main__":
    main()
