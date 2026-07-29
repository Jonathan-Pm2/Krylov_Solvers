#!/usr/bin/env python3
"""Phase-2 benchmark orchestrator — PROTOCOL.md sections 1.3, 3.3, 4, 6.

Drives the Julia (code/julia/harness.jl) and Python (code/python/harness.py)
harnesses across BOTH thread regimes and merges their output into schema-matched
CSVs under code/results/benchmarks/. It never regenerates data and never touches
the pre-existing code/results/*.csv.

What the runner adds on top of each harness invocation:
  * Thread regimes (1.3): sets OMP/OPENBLAS/MKL_NUM_THREADS in the child env and
    launches Julia with `-t <n>`; regime 1 = single controlled thread, regime 2 =
    matched physical-thread count in both languages. Every row is tagged with the
    regime and the requested thread count; the harness records the observed state.
  * Randomized execution order (3.3): the (regime, language, method, family,
    instance) work list is shuffled with a RECORDED deterministic seed. Rep-level
    interleaving is intentionally NOT done: reps live inside one process so the
    Julia warmup actually amortizes JIT (section 4); shuffling identical reps of a
    unit would be meaningless and re-launching per rep would re-contaminate timing.
  * Primary peak RSS (6): each unit subprocess is wrapped in `/usr/bin/time -v`.
    The runner parses "Maximum resident set size", subtracts a per-(regime,
    language) interpreter baseline (measured once), and writes it into
    peak_rss_bytes (peak_rss_method="usr_bin_time_v"). The harness's in-process
    VmHWM stays in peak_rss_inproc_bytes as a cross-check. If /usr/bin/time -v is
    unavailable the runner falls back to the harness in-process value and records
    that fact in the run metadata.

A run_metadata.json (seed, regime thread counts, BLAS backends, baselines,
shuffled work order, tool availability) is written next to the CSVs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(HERE, "bench_schema.json")
RESULTS_DIR = os.path.join(HERE, "results", "benchmarks")
JULIA_HARNESS = os.path.join(HERE, "julia", "harness.jl")
PY_HARNESS = os.path.join(HERE, "python", "harness.py")
ENV_JSON = os.path.join(HERE, "results", "environment.json")

TIME_BIN = "/usr/bin/time"

# --- experimental design ---------------------------------------------------
# Phase-4 PREVIEW grid: a small 2-D Laplacian scaling ladder (n up to ~10k) for
# the SPD phase, and three EXISTING singular instances (reused at their current
# modest sizes) for the Drazin phase. Kept intentionally small so the full
# pipeline (sweep -> aggregation -> tables -> figures -> scaling fit) can be
# de-risked in tens of minutes before the final large-n sweep.
SPD_INSTANCES = [
    ("laplacian_2d_spd", "laplacian_2d_spd/m032_n00001024"),
    ("laplacian_2d_spd", "laplacian_2d_spd/m064_n00004096"),
    ("laplacian_2d_spd", "laplacian_2d_spd/m096_n00009216"),
]
DRAZIN_INSTANCES = [
    ("similarity_singular", "similarity_singular/ncore040_k3_n00000043"),
    ("coupled_singular", "coupled_singular/ncore040_k3_n00000043"),
    ("blockdiag_high_index_singular", "blockdiag_high_index_singular/ncore040_k5_n00000045"),
]
SPD_METHODS = ["cg", "pcg_jacobi", "pcg_ic0"]
DRAZIN_METHODS = ["gmres", "dgmres"]

# 3 seeded RHS per instance for the preview (subset of the 20 seeded RHS on disk).
PREVIEW_RHS_SEEDS = [1000, 1001, 1002]

BENCH_SPD = os.path.join(RESULTS_DIR, "bench_spd.csv")
BENCH_DRAZIN = os.path.join(RESULTS_DIR, "bench_drazin.csv")
SPD_FAMILIES = {fam for fam, _ in SPD_INSTANCES}


def master_for(family):
    """SPD families -> bench_spd.csv, singular families -> bench_drazin.csv."""
    return BENCH_SPD if family in SPD_FAMILIES else BENCH_DRAZIN


def load_schema():
    with open(SCHEMA_PATH) as fh:
        return json.load(fh)["columns"]


def time_v_available():
    return os.path.isfile(TIME_BIN) and os.access(TIME_BIN, os.X_OK)


def parse_time_v_rss_kb(stderr_text):
    """Return 'Maximum resident set size (kbytes)' from /usr/bin/time -v, or None."""
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr_text)
    return int(m.group(1)) if m else None


def child_env(threads):
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    return env


def julia_cmd(threads, extra):
    return ["julia", f"-t{threads}", JULIA_HARNESS] + extra


def python_cmd(threads, extra):
    return [sys.executable, PY_HARNESS] + extra


def run_wrapped(cmd, env, use_time_v):
    """Run cmd, optionally under /usr/bin/time -v. Returns (stdout, external_rss_kb)."""
    if use_time_v:
        full = [TIME_BIN, "-v"] + cmd
    else:
        full = cmd
    proc = subprocess.run(full, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        sys.stderr.write(f"\n[FAILED] {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}\n")
        raise SystemExit(proc.returncode)
    rss_kb = parse_time_v_rss_kb(proc.stderr) if use_time_v else None
    return proc.stdout, rss_kb


def measure_baseline(lang, threads, use_time_v):
    """Interpreter/runtime baseline RSS (external kB and in-process bytes)."""
    extra = ["--mode", "baseline"]
    cmd = julia_cmd(threads, extra) if lang == "julia" else python_cmd(threads, extra)
    stdout, rss_kb = run_wrapped(cmd, child_env(threads), use_time_v)
    m = re.search(r"BASELINE_RSS_BYTES=(\d+)", stdout)
    inproc = int(m.group(1)) if m else 0
    ext = (rss_kb * 1024) if rss_kb is not None else inproc
    return ext, inproc


def merge_unit_csv(temp_path, master_path, cols, ext_peak_bytes, ext_baseline_bytes,
                   use_time_v):
    """Append the temp per-unit rows to the master CSV, injecting the primary
    /usr/bin/time -v peak RSS (baseline-subtracted). Returns the header line."""
    with open(temp_path, newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames
        rows = list(reader)
    new_master = (not os.path.isfile(master_path)) or os.path.getsize(master_path) == 0
    os.makedirs(os.path.dirname(master_path), exist_ok=True)
    with open(master_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new_master:
            w.writeheader()
        for row in rows:
            if use_time_v and ext_peak_bytes is not None:
                peak = max(ext_peak_bytes - ext_baseline_bytes, 0)
                nnz = float(row["nnz"])
                row["peak_rss_bytes"] = peak
                row["peak_rss_method"] = "usr_bin_time_v"
                row["baseline_rss_bytes"] = ext_baseline_bytes
                row["peak_bytes_per_nonzero"] = peak / nnz if nnz else "NaN"
            w.writerow(row)
    return ",".join(header)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=10, help="timed reps per unit")
    p.add_argument("--warmup", type=int, default=3,
                   help="warmup solves discarded (larger for the Julia JIT, section 4)")
    p.add_argument("--seed", type=int, default=20260727)
    p.add_argument("--rhs", type=int, default=3,
                   help="number of seeded RHS per instance (preview subset, max 3)")
    p.add_argument("--matched-threads", type=int, default=4,
                   help="thread count for the matched multi-thread regime")
    p.add_argument("--krylov-m", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--regimes", default="single_thread")
    p.add_argument("--langs", default="julia,python")
    p.add_argument("--fresh", action="store_true",
                   help="delete existing bench_*.csv before running")
    args = p.parse_args()

    cols = load_schema()
    use_time_v = time_v_available()
    langs = args.langs.split(",")
    regimes = args.regimes.split(",")
    regime_threads = {"single_thread": 1, "multi_thread": args.matched_threads}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rhs_seeds = PREVIEW_RHS_SEEDS[:args.rhs]
    if args.fresh:
        for pth in (BENCH_SPD, BENCH_DRAZIN):
            if os.path.isfile(pth):
                os.remove(pth)

    # --- baselines per (regime, language) ---------------------------------
    baselines = {}
    for regime in regimes:
        t = regime_threads[regime]
        for lang in langs:
            ext, inproc = measure_baseline(lang, t, use_time_v)
            baselines[(regime, lang)] = {"ext_bytes": ext, "inproc_bytes": inproc}

    # --- build + shuffle the work list (3.3) -------------------------------
    # One work item = one experimental unit = (regime, language, family, instance,
    # method, RHS-seed); the harness then runs warmup + args.reps inside it.
    work = []
    for regime in regimes:
        for lang in langs:
            for fam, key in SPD_INSTANCES:
                for method in SPD_METHODS:
                    for rhs_seed in rhs_seeds:
                        work.append((regime, lang, fam, key, method, rhs_seed))
            for fam, key in DRAZIN_INSTANCES:
                for method in DRAZIN_METHODS:
                    for rhs_seed in rhs_seeds:
                        work.append((regime, lang, fam, key, method, rhs_seed))
    rng = random.Random(args.seed)
    rng.shuffle(work)

    # --- execute -----------------------------------------------------------
    headers_seen = {}
    tmpdir = tempfile.mkdtemp(prefix="bench_units_")
    t_start = time.time()
    for i, (regime, lang, fam, key, method, rhs_seed) in enumerate(work, 1):
        threads = regime_threads[regime]
        temp_csv = os.path.join(tmpdir, f"u{i}_{lang}.csv")
        extra = [
            "--mode", "unit",
            "--regime", regime,
            "--requested-threads", str(threads),
            "--family", fam,
            "--instance-key", key,
            "--method", method,
            "--reps", str(args.reps),
            "--warmup", str(args.warmup),
            "--seed", str(args.seed),
            "--rhs-seed", str(rhs_seed),
            "--out", temp_csv,
            "--krylov-m", str(args.krylov_m),
            "--tol", repr(args.tol),
        ]
        cmd = julia_cmd(threads, extra) if lang == "julia" else python_cmd(threads, extra)
        print(f"[{i}/{len(work)}] {lang:6s} {regime:13s} {method:12s} rhs={rhs_seed} {key}")
        stdout, rss_kb = run_wrapped(cmd, child_env(threads), use_time_v)
        sys.stdout.write("    " + stdout.strip().replace("\n", "\n    ") + "\n")
        ext_peak = (rss_kb * 1024) if rss_kb is not None else None
        base = baselines[(regime, lang)]["ext_bytes"]
        header = merge_unit_csv(temp_csv, master_for(fam), cols, ext_peak, base, use_time_v)
        headers_seen.setdefault(lang, header)

    # --- run metadata ------------------------------------------------------
    meta = {
        "protocol_sections": ["1.3", "3.3", "4", "5", "6", "8.2"],
        "seed": args.seed,
        "reps": args.reps,
        "warmup_discarded": args.warmup,
        "krylov_m": args.krylov_m,
        "tol": args.tol,
        "regimes": {r: {"requested_threads": regime_threads[r]} for r in regimes},
        "languages": langs,
        "usr_bin_time_v_available": use_time_v,
        "usr_bin_time_v_note": None if use_time_v else
            "/usr/bin/time -v unavailable; peak_rss_bytes falls back to in-process VmHWM",
        "python_thread_control_note": (
            "threadpoolctl not installed; BLAS thread count is controlled only via "
            "OPENBLAS/OMP/MKL_NUM_THREADS env vars and observed from them"),
        "rhs_seeds": rhs_seeds,
        "baselines_bytes": {f"{r}|{l}": baselines[(r, l)] for r in regimes for l in langs},
        "work_order": [
            {"regime": r, "language": l, "family": f, "instance_key": k,
             "method": me, "rhs_seed": rs}
            for (r, l, f, k, me, rs) in work
        ],
        "schema_columns": cols,
        "elapsed_seconds": round(time.time() - t_start, 2),
        "csv_files": {"spd": os.path.relpath(BENCH_SPD, HERE),
                      "drazin": os.path.relpath(BENCH_DRAZIN, HERE)},
    }
    with open(os.path.join(RESULTS_DIR, "run_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # --- schema-identity check across languages ----------------------------
    canonical = ",".join(cols)
    print("\n=== schema check ===")
    ok = True
    for lang, h in headers_seen.items():
        match = (h == canonical)
        ok = ok and match
        print(f"  {lang}: header {'MATCHES' if match else 'DIFFERS from'} bench_schema.json")
    if headers_seen and len(set(headers_seen.values())) == 1:
        print("  julia/python headers are byte-identical")
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"\nDone in {meta['elapsed_seconds']}s. CSVs in {os.path.relpath(RESULTS_DIR, HERE)}/")
    if not ok:
        raise SystemExit("schema mismatch detected")


if __name__ == "__main__":
    main()
