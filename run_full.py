#!/usr/bin/env python3
"""Phase-4 FULL benchmark sweep — PROTOCOL.md sections 1.3, 3.3, 4, 5, 6, 10.

Generalizes run_benchmarks.py (which was hardcoded to the small Phase-4 preview
grid) to the full manifest. It enumerates EVERY instance from
code/data/manifest.json, assigns the SPD methods {cg, pcg_jacobi, pcg_ic0} to the
SPD families and {gmres, dgmres} to the singular families, and drives the two
harnesses (code/julia/harness.jl, code/python/harness.py) across two DECOUPLED
passes:

  * TIMING pass    : --reps 30 --warmup 5, 3 RHS, regimes single_thread AND
                     multi_thread (matched physical-core count). Measures
                     wall-clock; few RHS is fine because timing is ~RHS-independent.
                     -> bench_spd_timing.csv / bench_drazin_timing.csv
  * CONVERGENCE    : --reps 1 --warmup 0, up to 20 RHS, single_thread only, both
    pass             languages. Characterizes iteration-count spread across RHS
                     (PROTOCOL.md 3.2/3.3). Convergence is thread-independent so a
                     single regime suffices.
                     -> bench_spd_conv.csv / bench_drazin_conv.csv

The two passes write to SEPARATE CSVs (rather than adding a schema column) so the
canonical bench_schema.json is unchanged and Julia/Python rows still concatenate.

EFFICIENCY (amortize the language process startup): one harness process now runs
ALL requested RHS seeds for a given (instance, regime, language, method) via the
harness --rhs-seeds flag. Methods are kept one-per-process on purpose: peak RSS is
a per-process high-water mark (/usr/bin/time -v, /proc VmHWM), so mixing methods
in one process would attribute the max-memory method's footprint to every row and
corrupt the section-6 memory metric. Amortizing across RHS (3x in timing, up to
20x in convergence) captures the dominant Julia-startup cost while keeping peak
RSS a clean per-method measurement.

Everything else (thread regimes 1.3, randomized order 3.3, /usr/bin/time -v peak
RSS with baseline subtraction 6, incremental per-unit append so partial progress
survives) mirrors run_benchmarks.py.
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
MANIFEST = os.path.join(HERE, "data", "manifest.json")
TIME_BIN = "/usr/bin/time"

SPD_METHODS = ["cg", "pcg_jacobi", "pcg_ic0"]
DRAZIN_METHODS = ["gmres", "dgmres"]


# --- harness plumbing (mirrors run_benchmarks.py) --------------------------
def load_schema():
    with open(SCHEMA_PATH) as fh:
        return json.load(fh)["columns"]


def time_v_available():
    return os.path.isfile(TIME_BIN) and os.access(TIME_BIN, os.X_OK)


def parse_time_v_rss_kb(stderr_text):
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


def run_wrapped(cmd, env, use_time_v, allow_fail=False):
    full = ([TIME_BIN, "-v"] + cmd) if use_time_v else cmd
    proc = subprocess.run(full, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        sys.stderr.write(f"\n[FAILED] {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}\n")
        # A single failing unit (e.g. IC(0) breakdown on an ill-conditioned real
        # matrix) must NOT abort the whole sweep. Callers running units pass
        # allow_fail=True: the failure is logged and recorded, and the sweep
        # continues. Baselines keep the hard-fail behavior (allow_fail=False).
        if allow_fail:
            return None, None
        raise SystemExit(proc.returncode)
    rss_kb = parse_time_v_rss_kb(proc.stderr) if use_time_v else None
    return proc.stdout, rss_kb


def measure_baseline(lang, threads, use_time_v):
    extra = ["--mode", "baseline"]
    cmd = julia_cmd(threads, extra) if lang == "julia" else python_cmd(threads, extra)
    stdout, rss_kb = run_wrapped(cmd, child_env(threads), use_time_v)
    m = re.search(r"BASELINE_RSS_BYTES=(\d+)", stdout)
    inproc = int(m.group(1)) if m else 0
    ext = (rss_kb * 1024) if rss_kb is not None else inproc
    return ext, inproc


def merge_unit_csv(temp_path, master_path, cols, ext_peak_bytes, ext_baseline_bytes,
                   use_time_v):
    """Append the per-unit rows to the master CSV, injecting the /usr/bin/time -v
    peak RSS (baseline-subtracted). One process = one (instance,regime,lang,method)
    running all its RHS seeds; all rows share that method's peak RSS."""
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
    return ",".join(header), len(rows)


# --- experimental design from the manifest ---------------------------------
def is_singular(family):
    return "singular" in family


def instance_rhs_seeds(inst):
    """Available RHS seeds for an instance: the seeds in its rhs_set (sorted), or
    [None] (primary RHS, seed 0) for Phase-1 instances with no seeded set."""
    rs = inst.get("rhs_set")
    if rs:
        return [int(e["seed"]) for e in rs]
    return [None]


def enumerate_units(manifest, n_rhs, reps, warmup):
    """One unit per (family, instance_key, method) with its capped RHS-seed list
    and its (per-unit) reps/warmup. n_rhs caps how many of the instance's available
    seeds are used; reps/warmup start at the pass defaults and may be lowered by the
    cap policy on the few superlinear cells."""
    units = []
    for inst in manifest["instances"]:
        fam = inst["family"]
        key = os.path.dirname(inst["matrix_file"])
        methods = DRAZIN_METHODS if is_singular(fam) else SPD_METHODS
        seeds = instance_rhs_seeds(inst)[:n_rhs]
        for method in methods:
            units.append({
                "family": fam, "instance_key": key, "method": method,
                "n": int(inst["n"]), "nnz": int(inst["nnz"]),
                "rhs_seeds": list(seeds), "reps": reps, "warmup": warmup,
                "phase": "drazin" if is_singular(fam) else "spd",
            })
    return units


# Dense-cell threshold: only the similarity_singular instances at n>=1000 exceed
# this nnz (1e6/4e6/25e6); every other family stays far below it.
DENSE_NNZ = 500_000
LARGE_SPD_N = 20_000  # lap2d n=24964,50176 and lap3d n=27000

# HARD DROP (logged): the dense similarity_singular matrices are effectively dense
# (nnz ~ n^2), and DGMRES forms A^a r every iteration (O(a*nnz*iters^2)); at
# n=2000 one solve is ~18-40 s and at n=5000 ~2 min, so a repeated timing sweep is
# hours per cell. These two sizes (n=2000, n=5000) are DROPPED from both passes for
# the singular methods. The sparse SPD Laplacian ladder (up to n=50176) is cheap
# and fully covered; the singular DGMRES/GMRES sweep is capped at n<=1000, which
# still leaves a multi-n singular ladder (n = 42..1000) for the section-10 fit.
DROP_SINGULAR_N = 2000


def drop_intractable(units, log):
    """Remove (and log) the intractable dense-singular cells. No silent truncation:
    every dropped (family, instance, method, n) is recorded and returned."""
    kept, dropped = [], []
    for u in units:
        if u["phase"] == "drazin" and u["n"] >= DROP_SINGULAR_N:
            dropped.append({"family": u["family"], "instance_key": u["instance_key"],
                            "method": u["method"], "n": u["n"], "nnz": u["nnz"],
                            "reason": (f"dense singular (nnz={u['nnz']}~n^2); DGMRES "
                                       f"A^a r is O(a*nnz*iters^2), ~minutes/solve; "
                                       f"dropped at n>={DROP_SINGULAR_N}")})
            log(f"  [DROP] {u['method']:6s} n={u['n']:>6} {u['instance_key']}: "
                f"nnz={u['nnz']} -> dropped (dense; intractable repeated-timing cost)")
        else:
            kept.append(u)
    return kept, dropped


def apply_caps(units, tag, cap_hours, log):
    """Intelligent, LOGGED coverage caps (PROTOCOL forbids silent truncation).
    The only superlinear cost is DGMRES on the dense similarity_singular instances
    (its A^a*V formation is O(a*nnz*iters^2)); a full reps=30 x 3-RHS sweep there
    would take many hours. We cap ONLY those cells (plus an RHS drop on the largest
    SPD cells, whose timing is RHS-independent). Returns a list of cap-log records."""
    capped = []

    def note(u, what, why):
        rec = {"tag": tag, "family": u["family"], "instance_key": u["instance_key"],
               "method": u["method"], "n": u["n"], "nnz": u["nnz"],
               "change": what, "reason": why}
        capped.append(rec)
        log(f"  [CAP {tag}] {u['method']:6s} n={u['n']:>6} {u['instance_key']}: "
            f"{what}  ({why})")

    for u in units:
        dense_sing = u["phase"] == "drazin" and u["nnz"] >= DENSE_NNZ
        if tag == "timing":
            if u["phase"] == "spd" and u["n"] >= LARGE_SPD_N and len(u["rhs_seeds"]) > 1:
                orig = len(u["rhs_seeds"])
                u["rhs_seeds"] = u["rhs_seeds"][:1]
                note(u, f"RHS {orig}->1",
                     "timing is RHS-independent (PROTOCOL 4); largest SPD cells")
            if dense_sing:
                if len(u["rhs_seeds"]) > 1:
                    orig = len(u["rhs_seeds"])
                    u["rhs_seeds"] = u["rhs_seeds"][:1]
                    note(u, f"RHS {orig}->1", "dense singular cell; timing RHS-independent")
                if u["method"] == "dgmres":
                    if u["n"] >= 5000:
                        u["reps"], u["warmup"] = 4, 1
                    elif u["n"] >= 2000:
                        u["reps"], u["warmup"] = 8, 2
                    elif u["n"] >= 1000:
                        u["reps"], u["warmup"] = 20, 3
                    note(u, f"reps->{u['reps']} warmup->{u['warmup']}",
                         "DGMRES A^a*V is O(a*nnz*iters^2); ~18-120 s/solve here")
        elif tag == "conv":
            if dense_sing and u["method"] == "dgmres":
                orig = len(u["rhs_seeds"])
                if u["n"] >= 5000:
                    u["rhs_seeds"] = u["rhs_seeds"][:3]
                elif u["n"] >= 2000:
                    u["rhs_seeds"] = u["rhs_seeds"][:5]
                elif u["n"] >= 1000:
                    u["rhs_seeds"] = u["rhs_seeds"][:10]
                note(u, f"RHS {orig}->{len(u['rhs_seeds'])}",
                     "representative RHS subset; each DGMRES solve is 4-120 s")
    return capped


def master_for(phase, tag):
    name = f"bench_{phase}_{tag}.csv"
    return os.path.join(RESULTS_DIR, name)


def completed_keys(tag):
    """Set of (regime, language, family, instance_key, method) already present in
    this pass's CSVs. Lets a restart RESUME (skip finished units) instead of
    redoing them or, worse, appending duplicates. Units are written per-completed
    process, so partial/in-flight units never appear here."""
    done = set()
    for phase in ("spd", "drazin"):
        path = master_for(phase, tag)
        if not os.path.isfile(path):
            continue
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                done.add((row["regime"], row["language"], row["family"],
                          row["instance_key"], row["method"]))
    return done


def seeds_arg(seeds):
    """--rhs-seeds string; None -> omit (harness uses the primary RHS)."""
    concrete = [s for s in seeds if s is not None]
    return ",".join(str(s) for s in concrete) if concrete else None


# --- one pass ---------------------------------------------------------------
def build_extra(unit, regime, threads, reps, warmup, seed, krylov_m, tol, out, seeds):
    extra = [
        "--mode", "unit", "--regime", regime,
        "--requested-threads", str(threads),
        "--family", unit["family"], "--instance-key", unit["instance_key"],
        "--method", unit["method"], "--reps", str(reps), "--warmup", str(warmup),
        "--seed", str(seed), "--out", out,
        "--krylov-m", str(krylov_m), "--tol", repr(tol),
    ]
    sa = seeds_arg(seeds)
    if sa is not None:
        extra += ["--rhs-seeds", sa]
    return extra


def run_one_process(unit, regime, threads, lang, args, baselines, use_time_v,
                    cols, tmpdir, idx):
    temp_csv = os.path.join(tmpdir, f"u{idx}_{lang}.csv")
    seeds = unit["rhs_seeds"]
    extra = build_extra(unit, regime, threads, unit["reps"], unit["warmup"],
                        args["seed"], args["krylov_m"], args["tol"], temp_csv, seeds)
    cmd = julia_cmd(threads, extra) if lang == "julia" else python_cmd(threads, extra)
    t0 = time.time()
    stdout, rss_kb = run_wrapped(cmd, child_env(threads), use_time_v, allow_fail=True)
    dt = time.time() - t0
    if stdout is None:            # unit failed -> header None signals a skip
        return dt, None, 0, None
    ext_peak = (rss_kb * 1024) if rss_kb is not None else None
    base = baselines[(regime, lang)]["ext_bytes"]
    master = master_for(unit["phase"], args["tag"])
    header, nrows = merge_unit_csv(temp_csv, master, cols, ext_peak, base, use_time_v)
    return dt, header, nrows, stdout


def run_pass(tag, regimes, args, units, cols, use_time_v, log):
    """Execute one pass (timing or conv) over the (already capped) unit list."""
    langs = args["langs"]
    matched = args["matched_threads"]
    regime_threads = {"single_thread": 1, "multi_thread": matched}

    # baselines per (regime, lang)
    baselines = {}
    for regime in regimes:
        t = regime_threads[regime]
        for lang in langs:
            ext, inproc = measure_baseline(lang, t, use_time_v)
            baselines[(regime, lang)] = {"ext_bytes": ext, "inproc_bytes": inproc}

    # full work list = (regime, lang, unit); RHS seeds loop inside each process
    work = []
    for regime in regimes:
        for lang in langs:
            for u in units:
                work.append((regime, lang, u))
    rng = random.Random(args["seed"])
    rng.shuffle(work)

    # RESUME: skip units already recorded in this pass's CSVs (survives restarts).
    done = completed_keys(tag)
    if done:
        before = len(work)
        work = [(r, l, u) for (r, l, u) in work
                if (r, l, u["family"], u["instance_key"], u["method"]) not in done]
        log(f"  [RESUME] {before - len(work)} of {before} units already in CSV; "
            f"running the remaining {len(work)}.")

    log(f"\n=== PASS '{tag}': {len(work)} processes "
        f"(units={len(units)} x regimes={len(regimes)} x langs={len(langs)}), "
        f"n_rhs<={args['n_rhs']} ===")

    headers_seen = {}
    failed = []
    tmpdir = tempfile.mkdtemp(prefix=f"bench_{tag}_")
    t_start = time.time()
    total_rows = 0
    per_lang_seconds = {}
    for i, (regime, lang, u) in enumerate(work, 1):
        threads = regime_threads[regime]
        seeds = u["rhs_seeds"]
        dt, header, nrows, stdout = run_one_process(
            u, regime, threads, lang, args, baselines, use_time_v, cols, tmpdir, i)
        per_lang_seconds.setdefault(lang, 0.0)
        per_lang_seconds[lang] += dt
        if header is None:            # unit failed: log, record, and continue
            failed.append({"regime": regime, "language": lang, "family": u["family"],
                           "instance_key": u["instance_key"], "method": u["method"],
                           "n": u["n"]})
            log(f"[{tag} {i}/{len(work)}] {lang:6s} {regime:13s} {u['method']:11s} "
                f"n={u['n']:>7} FAILED, skipped (see [FAILED] above)  {u['instance_key']}")
            continue
        total_rows += nrows
        headers_seen.setdefault(lang, header)
        log(f"[{tag} {i}/{len(work)}] {lang:6s} {regime:13s} {u['method']:11s} "
            f"n={u['n']:>7} rhs={len(seeds)} r={u['reps']} {dt:7.1f}s  {u['instance_key']}")

    elapsed = time.time() - t_start
    shutil.rmtree(tmpdir, ignore_errors=True)

    canonical = ",".join(cols)
    schema_ok = all(h == canonical for h in headers_seen.values())
    return {
        "tag": tag, "regimes": regimes, "n_processes": len(work),
        "n_units": len(units), "n_rows": total_rows,
        "n_failed": len(failed), "failed": failed,
        "elapsed_seconds": round(elapsed, 1),
        "per_lang_seconds": {k: round(v, 1) for k, v in per_lang_seconds.items()},
        "schema_ok": schema_ok, "baselines": baselines,
    }


# --- a-priori estimate (PROTOCOL: sample before the big loop) ---------------
def estimate_pass(tag, regimes, args, units, cols, use_time_v, log):
    """Time a few representative (capped) units spanning the n range per language,
    derive a per-solve cost, and project the full-pass wall clock. Returns the
    projected seconds. Sample rows are written to a scratch CSV, never the real one."""
    langs = args["langs"]

    baselines = {}
    for lang in langs:
        ext, inproc = measure_baseline(lang, 1, use_time_v)
        baselines[("single_thread", lang)] = {"ext_bytes": ext, "inproc_bytes": inproc}

    # representative units: for singular prefer DGMRES (the expensive method).
    def pick(phase):
        pool = [u for u in units if u["phase"] == phase]
        if phase == "drazin":
            pool = [u for u in pool if u["method"] == "dgmres"] or pool
        us = sorted(pool, key=lambda u: u["n"])
        if not us:
            return []
        idxs = sorted(set([0, len(us) // 2, len(us) - 1]))
        return [us[j] for j in idxs]

    samples = pick("spd") + pick("drazin")
    log(f"\n--- estimating pass '{tag}' from {len(samples)} sample units "
        f"(single_thread, {langs}) ---")
    tmpdir = tempfile.mkdtemp(prefix=f"est_{tag}_")
    # per-solve seconds keyed by lang -> list of (n, phase, sec_per_solve)
    pts = {lang: [] for lang in langs}
    STARTUP = {"julia": 4.0, "python": 0.4}  # rough per-process startup to net out
    # A cheap probe: 1 RHS, a couple of solves is enough to read per-solve cost;
    # the full capped reps would make the largest DGMRES probe take ~20 min.
    PROBE_REPS, PROBE_WARMUP = 2, 0
    for j, u in enumerate(samples, 1):
        seeds1 = u["rhs_seeds"][:1]
        nsolve = max(len(seeds1) * (PROBE_REPS + PROBE_WARMUP), 1)
        for lang in langs:
            # write samples to a scratch out so they never touch the real CSVs
            temp = os.path.join(tmpdir, f"est{j}_{lang}.csv")
            extra = build_extra(u, "single_thread", 1, PROBE_REPS, PROBE_WARMUP,
                                args["seed"], args["krylov_m"], args["tol"], temp,
                                seeds1)
            cmd = julia_cmd(1, extra) if lang == "julia" else python_cmd(1, extra)
            t0 = time.time()
            run_wrapped(cmd, child_env(1), use_time_v)
            dt = time.time() - t0
            per_solve = max(dt - STARTUP[lang], 0.01) / nsolve
            pts[lang].append((u["n"], u["phase"], per_solve))
            log(f"  sample {lang:6s} {u['method']:11s} n={u['n']:>7} "
                f"phase={u['phase']:6s} {nsolve:>3} solves -> {dt:7.1f}s "
                f"({per_solve*1000:.1f} ms/solve)")
    shutil.rmtree(tmpdir, ignore_errors=True)

    def per_solve(lang, phase, n):
        cand = [(abs(np_ - n), ps) for (np_, ph, ps) in pts[lang] if ph == phase]
        if not cand:
            cand = [(abs(np_ - n), ps) for (np_, ph, ps) in pts[lang]]
        return min(cand)[1] if cand else 0.0

    total = 0.0
    for regime in regimes:
        for lang in langs:
            for u in units:
                nsolve = len(u["rhs_seeds"]) * (u["reps"] + u["warmup"])
                total += STARTUP[lang] + per_solve(lang, u["phase"], u["n"]) * nsolve
    log(f"  -> projected pass '{tag}' wall clock ~ {total/60:.1f} min "
        f"({total/3600:.2f} h) over {len(units)*len(regimes)*len(langs)} processes")
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pass", dest="which", default="both",
                   choices=["timing", "conv", "both"])
    p.add_argument("--seed", type=int, default=20260727)
    p.add_argument("--matched-threads", type=int, default=8,
                   help="physical-core count for the matched multi-thread regime")
    p.add_argument("--krylov-m", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--langs", default="julia,python")
    p.add_argument("--estimate-only", action="store_true")
    # per-pass knobs (defaults follow the task spec)
    p.add_argument("--timing-reps", type=int, default=30)
    p.add_argument("--timing-warmup", type=int, default=5)
    p.add_argument("--timing-rhs", type=int, default=3)
    p.add_argument("--conv-reps", type=int, default=1)
    p.add_argument("--conv-warmup", type=int, default=0)
    p.add_argument("--conv-rhs", type=int, default=20)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--cap-hours", type=float, default=2.0)
    p.add_argument("--skip-estimate", action="store_true",
                   help="skip the a-priori sample estimate (costs are already known)")
    args = p.parse_args()

    cols = load_schema()
    use_time_v = time_v_available()
    langs = args.langs.split(",")
    with open(MANIFEST) as fh:
        manifest = json.load(fh)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log_lines = []
    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    pass_specs = {
        "timing": dict(tag="timing", regimes=["single_thread", "multi_thread"],
                       reps=args.timing_reps, warmup=args.timing_warmup,
                       n_rhs=args.timing_rhs),
        "conv": dict(tag="conv", regimes=["single_thread"],
                     reps=args.conv_reps, warmup=args.conv_warmup,
                     n_rhs=args.conv_rhs),
    }
    todo = ["timing", "conv"] if args.which == "both" else [args.which]

    if args.fresh:
        for name in ("spd_timing", "drazin_timing", "spd_conv", "drazin_conv"):
            pth = os.path.join(RESULTS_DIR, f"bench_{name}.csv")
            if os.path.isfile(pth):
                os.remove(pth)

    summaries = {}
    for which in todo:
        spec = pass_specs[which]
        common = dict(seed=args.seed, matched_threads=args.matched_threads,
                      krylov_m=args.krylov_m, tol=args.tol, langs=langs,
                      tag=spec["tag"], n_rhs=spec["n_rhs"])

        # enumerate -> drop intractable dense-singular cells -> apply cap policy.
        units = enumerate_units(manifest, spec["n_rhs"], spec["reps"], spec["warmup"])
        log(f"\n--- pass '{spec['tag']}': dropping intractable dense-singular cells ---")
        units, dropped = drop_intractable(units, log)
        capped = apply_caps(units, spec["tag"], args.cap_hours, log)
        proj = 0.0
        if not args.skip_estimate:
            proj = estimate_pass(spec["tag"], spec["regimes"], common, units, cols,
                                 use_time_v, log)
            if proj > args.cap_hours * 3600:
                log(f"  [NOTE] pass '{spec['tag']}' still projects {proj/3600:.2f} h > "
                    f"{args.cap_hours} h after caps; proceeding (all caps logged).")
        if args.estimate_only:
            summaries[which] = {"projected_hours": round(proj / 3600, 3),
                                "n_capped_cells": len(capped),
                                "n_dropped_cells": len(dropped),
                                "dropped": dropped, "capped": capped}
            continue

        summary = run_pass(spec["tag"], spec["regimes"], common, units, cols,
                           use_time_v, log)
        summary["projected_hours"] = round(proj / 3600, 3)
        summary["capped"] = capped
        summary["dropped"] = dropped
        summaries[which] = summary
        log(f"=== pass '{spec['tag']}' done in {summary['elapsed_seconds']}s "
            f"({summary['elapsed_seconds']/3600:.2f} h); rows={summary['n_rows']}; "
            f"schema_ok={summary['schema_ok']} ===")

    meta = {
        "protocol_sections": ["1.3", "3.3", "4", "5", "6", "8.2", "10"],
        "seed": args.seed, "matched_threads": args.matched_threads,
        "krylov_m": args.krylov_m, "tol": args.tol, "langs": langs,
        "usr_bin_time_v_available": use_time_v,
        "pass_specs": {k: pass_specs[k] for k in todo},
        "summaries": {k: {kk: vv for kk, vv in v.items() if kk != "baselines"}
                      for k, v in summaries.items()},
        "log": log_lines,
    }
    with open(os.path.join(RESULTS_DIR, "run_full_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"\nWrote run_full_metadata.json. CSVs under {os.path.relpath(RESULTS_DIR, HERE)}/")


if __name__ == "__main__":
    main()
