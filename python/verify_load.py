"""Python half of the cross-language bit-for-bit verification (PROTOCOL.md section 2).

Loads every instance listed in code/data/manifest.json from disk (never
regenerates) and asserts:

  1. Self-consistency vs the manifest recorded at generation time:
       - matrix A: shape, nnz, and canonical value checksum match;
       - RHS   b: length and raw-bytes checksum match.
  2. Direct cross-language agreement vs the Julia loader's dumps
     (code/data/_verify_julia/, produced by code/julia/verify_load.jl):
       - A: Julia-loaded shape/nnz/value-checksum == Python-loaded == manifest;
       - b: max_i |b_i^Python - b_i^Julia| == 0.0  (bitwise identical).

Any mismatch prints the offending instance and exits with a nonzero status
(fail loudly). Run AFTER code/julia/verify_load.jl.

Run:  python code/python/verify_load.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_layer import read_mtx, read_npy, matrix_values_sha256, vector_sha256  # noqa: E402

DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))
DUMP_DIR = os.path.join(DATA_DIR, "_verify_julia")


def main() -> int:
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"manifest not found: {manifest_path} (run generate_data.jl first)")
        return 2
    with open(manifest_path) as fh:
        manifest = json.load(fh)

    failures = []
    max_b_diff = 0.0

    for inst in manifest["instances"]:
        key = os.path.dirname(inst["matrix_file"])
        A = read_mtx(os.path.join(DATA_DIR, inst["matrix_file"]))
        b = read_npy(os.path.join(DATA_DIR, inst["rhs_file"]))

        # --- assert A against the manifest -------------------------------
        got_shape = [A.shape[0], A.shape[1]]
        if got_shape != list(inst["shape"]):
            failures.append(f"{key}: A shape {got_shape} != manifest {inst['shape']}")
        got_nnz = int(A.nnz)
        if got_nnz != inst["nnz"]:
            failures.append(f"{key}: A nnz {got_nnz} != manifest {inst['nnz']}")
        amv = matrix_values_sha256(A)
        if amv != inst["matrix_values_sha256"]:
            failures.append(
                f"{key}: A checksum mismatch (loaded {amv} != manifest {inst['matrix_values_sha256']})"
            )

        # --- assert b against the manifest -------------------------------
        if len(b) != inst["rhs_len"]:
            failures.append(f"{key}: |b| {len(b)} != manifest {inst['rhs_len']}")
        bv = vector_sha256(b)
        if bv != inst["rhs_sha256"]:
            failures.append(
                f"{key}: b checksum mismatch (loaded {bv} != manifest {inst['rhs_sha256']})"
            )

        # --- cross-language check vs the Julia loader --------------------
        dkey = os.path.join(DUMP_DIR, key)
        info_path = os.path.join(dkey, "info.json")
        b_jl_path = os.path.join(dkey, "b_loaded.npy")
        if not (os.path.isfile(info_path) and os.path.isfile(b_jl_path)):
            failures.append(f"{key}: Julia cross-check dump missing (run verify_load.jl first)")
            continue
        with open(info_path) as fh:
            jl = json.load(fh)
        if list(jl["shape"]) != got_shape:
            failures.append(f"{key}: Julia-loaded A shape {jl['shape']} != Python {got_shape}")
        if int(jl["nnz"]) != got_nnz:
            failures.append(f"{key}: Julia-loaded A nnz {jl['nnz']} != Python {got_nnz}")
        if jl["matrix_values_sha256"] != amv:
            failures.append(
                f"{key}: A checksum Julia {jl['matrix_values_sha256']} != Python {amv}"
            )
        b_jl = read_npy(b_jl_path)
        if len(b_jl) != len(b):
            failures.append(f"{key}: b length Julia {len(b_jl)} != Python {len(b)}")
        else:
            d = float(np.max(np.abs(b - b_jl))) if len(b) else 0.0
            max_b_diff = max(max_b_diff, d)
            if d != 0.0:
                failures.append(f"{key}: max|b_python - b_julia| = {d!r} != 0")

        # --- Phase-3 extension: full rhs_set + x_star (section 3.3, 8.3) --
        # Self-consistency vs manifest AND bitwise cross-language agreement vs
        # the Julia loader dumps, for every seeded RHS and every x_star vector.
        for r in inst.get("rhs_set", []):
            bi = read_npy(os.path.join(DATA_DIR, r["rhs_file"]))
            if vector_sha256(bi) != r["rhs_sha256"]:
                failures.append(f"{key}: rhs {r['rhs_file']} checksum mismatch")
            bi_jl_path = os.path.join(dkey, f"b_seed{r['seed']}.npy")
            if os.path.isfile(bi_jl_path):
                d = float(np.max(np.abs(bi - read_npy(bi_jl_path)))) if len(bi) else 0.0
                max_b_diff = max(max_b_diff, d)
                if d != 0.0:
                    failures.append(f"{key}: rhs seed {r['seed']} max|py-jl| = {d!r} != 0")
            else:
                failures.append(f"{key}: Julia dump for rhs seed {r['seed']} missing")
            if r.get("x_star_file"):
                xi = read_npy(os.path.join(DATA_DIR, r["x_star_file"]))
                if vector_sha256(xi) != r["x_star_sha256"]:
                    failures.append(f"{key}: x_star {r['x_star_file']} checksum mismatch")
                xi_jl_path = os.path.join(dkey, f"x_star_seed{r['seed']}.npy")
                if os.path.isfile(xi_jl_path):
                    d = float(np.max(np.abs(xi - read_npy(xi_jl_path)))) if len(xi) else 0.0
                    max_b_diff = max(max_b_diff, d)
                    if d != 0.0:
                        failures.append(f"{key}: x_star seed {r['seed']} max|py-jl| = {d!r} != 0")
                else:
                    failures.append(f"{key}: Julia dump for x_star seed {r['seed']} missing")

    n = len(manifest["instances"])
    if not failures:
        print(f"Python verify_load: PASS — {n} instances match manifest and Julia loader "
              f"to bit precision.")
        print(f"  max_i |b_i^Python - b_i^Julia| over all instances = {max_b_diff!r}")
        return 0
    print("Python verify_load: FAIL")
    for f in failures:
        print("  -", f)
    print(f"  (max_i |b_i^Python - b_i^Julia| observed = {max_b_diff!r})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
