"""Python side of the language-neutral data layer (PROTOCOL.md section 2).

Python NEVER generates test data: it only *loads* the files written once by
code/julia/generate_data.jl and recomputes the canonical checksums so a
bit-for-bit round-trip can be asserted independently of Julia.

Loaders:
  - read_npy: RHS b from a NumPy .npy file (via numpy.load).
  - read_mtx: matrix A from a Matrix Market coordinate real `general` file,
    with a minimal parser (no implicit symmetric expansion) that mirrors the
    Julia reader exactly.

Checksums mirror code/julia/data_layer.jl byte-for-byte:
  - matrix_values_sha256: SHA-256 over canonical (col,row)-sorted triplets, each
    serialized as little-endian <i8 row, <i8 col, <f8 value (1-based indices).
  - vector_sha256: SHA-256 over the raw little-endian <f8 bytes of the vector.

Only stdlib + already-used deps (numpy, scipy) are required; scipy is imported
lazily and is optional (used only as an independent cross-check of read_mtx).
"""
from __future__ import annotations

import hashlib

import numpy as np
import scipy.sparse as sp


def read_npy(path: str) -> np.ndarray:
    """Load a 1-D float64 vector from a .npy file."""
    v = np.load(path)
    v = np.ascontiguousarray(v, dtype=np.float64).ravel()
    return v


def read_mtx(path: str) -> sp.csr_matrix:
    """Read a Matrix Market coordinate real `general` file into a CSR matrix.

    Mirrors code/julia/data_layer.jl:read_mtx: 1-based indices in the file are
    converted to 0-based; no symmetric expansion is performed.
    """
    rows, cols, vals = [], [], []
    shape = None
    declared_nnz = -1
    seen_size = False
    with open(path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if not seen_size:
                m, n, declared_nnz = int(parts[0]), int(parts[1]), int(parts[2])
                shape = (m, n)
                seen_size = True
            else:
                rows.append(int(parts[0]) - 1)
                cols.append(int(parts[1]) - 1)
                vals.append(float(parts[2]))
    if declared_nnz >= 0 and len(vals) != declared_nnz:
        raise ValueError(
            f"read_mtx: declared nnz {declared_nnz} != entries read {len(vals)} in {path}"
        )
    A = sp.csr_matrix(
        (np.asarray(vals, dtype=np.float64),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=shape,
    )
    return A


def canonical_triplets(A):
    """Stored entries sorted by (column, row), returned as 1-based int arrays.

    Mirrors the Julia canonical order so the value checksum is storage-order
    independent across languages.
    """
    coo = (A.tocoo() if sp.issparse(A) else sp.coo_matrix(A))
    coo = coo.copy()
    coo.sum_duplicates()
    I = coo.row.astype(np.int64) + 1   # 1-based, match Julia
    J = coo.col.astype(np.int64) + 1
    V = coo.data.astype(np.float64)
    order = np.lexsort((I, J))         # primary key = column (J), secondary = row (I)
    return I[order], J[order], V[order]


def matrix_values_sha256(A) -> str:
    """SHA-256 over canonical triplets: <i8 row, <i8 col, <f8 value, little-endian."""
    I, J, V = canonical_triplets(A)
    rec = np.empty(len(V), dtype=np.dtype([("i", "<i8"), ("j", "<i8"), ("v", "<f8")]))
    rec["i"] = I
    rec["j"] = J
    rec["v"] = V
    return hashlib.sha256(rec.tobytes()).hexdigest()


def vector_sha256(v: np.ndarray) -> str:
    """SHA-256 over raw little-endian <f8 bytes of the vector."""
    buf = np.ascontiguousarray(v, dtype="<f8")
    return hashlib.sha256(buf.tobytes()).hexdigest()
