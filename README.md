# Krylov Solvers in Julia and Python — Reproducibility Repository

Code, data, and manuscript for *"Benchmarking Krylov Solvers in Julia and Python
and Assessing the Numerical Reliability of DGMRES for Singular Systems"*
(Proaño Morales, Cuenca, Pozo Parra).

## Layout
- `julia/`, `python/` — mirrored numerical cores (CG, Jacobi-PCG, IC(0)-PCG, GMRES, DGMRES) and the benchmark harness.
- `data/` — byte-identical problem instances (Matrix Market matrices, IEEE-754 `.npy` right-hand sides) and `manifest.json`.
- `results/` — generated LaTeX tables and figures; `results/benchmarks/` holds the raw per-repetition CSVs; `results/environment.json` records exact versions.
- `main.tex` (modular; `\input`s `drazin_section.tex`, `spd_correction.tex`) and `manuscript_submission.tex` (flattened) — the manuscript.
- `PROTOCOL.md` — the experimental protocol governing the benchmark.

## Environment
Julia 1.12.5; Python 3.13.12 (numpy 2.4.4, scipy 1.17.1); OpenBLAS (ILP64). Full details in `results/environment.json`.

## Reproduce
1. Generate + verify data (bit-for-bit): `julia julia/generate_data.jl` then `python python/verify_load.py`
2. Run the benchmark sweep: `python run_full.py --pass both`
3. Regenerate tables/figures: `python python/aggregate.py && python python/make_tables_v2.py && python python/plot_v2.py`
4. Build the paper (from this directory): `pdflatex main && pdflatex main`

> The ~1 GB of generated problem instances under `data/` are **not** stored in this
> repository; they are regenerated deterministically (and verified byte-for-byte) by the
> generators listed above. `data/manifest.json` records every instance with SHA-256
> checksums.

## License

The code in this repository is released under the MIT License (see `LICENSE`).
The generated problem-instance dataset is not distributed in this repository; it is
regenerated deterministically by the code above and verified against the checksummed
`data/manifest.json`.
