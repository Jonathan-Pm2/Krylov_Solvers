# Aggregation path and variance components (round-2 note)

## Aggregation path

The headline solve time for one experimental unit (language, regime, method, family, instance, n) is formed hierarchically, NOT by pooling all repetitions as independent:

```
per-rep sample  --median over 30 reps-->  per-RHS value
per-RHS value   --median across seeds-->  UNIT HEADLINE
```

The 30 technical repetitions inside a fixed RHS seed are measurement replicates (same matrix, same right-hand side); the seeds are distinct right-hand sides. Collapsing reps first with a MEDIAN (not a mean), then collapsing RHS, keeps the two levels of the hierarchy separate and -- crucially -- makes the headline robust to the heavy-tailed repetition distribution documented below.

## Two variance components

For each unit we measure, on the raw per-rep samples:

- WITHIN-RHS spread: pooled standard deviation of the technical repetitions inside a fixed RHS seed (measurement / machine noise), as a fraction of the headline median (CV_rep).
- BETWEEN-RHS spread: standard deviation of the per-RHS medians across the seeded right-hand sides, as a fraction of the headline median (CV_RHS).

Over the 156 single-thread units with >= 2 RHS seeds:

- median CV_rep  (within-RHS): 10.1%
- median CV_RHS  (between-RHS): 1.7%
- units where between-RHS exceeds within-RHS: 26 / 156 (17%)

### The within-RHS spread is language-asymmetric

- median CV_rep  Julia: 107.8%   Python: 2.3%
- median CV_RHS  Julia: 3.6%   Python: 1.1%

The within-RHS repetition spread is dominated by Julia, where individual repetitions carry occasional very large timings (garbage-collection pauses and residual JIT effects that survive the discarded warm-up). These are rare heavy-tail events: they inflate the standard deviation far above the typical repetition, which is exactly why the standard-deviation-based CV_rep can exceed 100%. Python's repetition distribution is tight by comparison. Because the headline collapses reps with a MEDIAN, these tails do not distort the reported solve time -- this is the concrete justification for median-based aggregation.

Two consequences for the manuscript:

1. A confidence interval built from pooled repetitions treated as independent is not credible: it assumes (30 x #RHS) independent draws, ignores the nested structure, and -- for Julia -- is computed over a heavy-tailed distribution whose mean and variance are unstable. The effective number of independent draws is closer to #RHS.
2. The right-hand side is a genuine, non-negligible source of variation (median CV_RHS around 2%) and must be sampled (multiple seeds), not fixed to one vector.

See `variance_components.csv` for the full per-unit table and Table~\ref{tab:variance_components_v2} for representative units.
