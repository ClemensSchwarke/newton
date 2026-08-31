# Discretization, cost and stability of the reduced elastic leg in the drop test

`shank_parity.md` established how finely the PRDM spring chain must be discretized but did not
do the same for the reduced elastic blade, and its measurements predate the unified elastic
solve. This report covers both: how few substeps and iterations the elastic leg needs, and
what the unified arrangement costs against the split one.

Both legs are dropped in the same model under one solver. The PRDM chain is rigid and cannot
be affected by the elastic arrangement, so it serves as a control.

| arrangement | how the elastic frame is solved |
| --- | --- |
| unified (current) | frame and modes together in one `(6 + n_m)` block |
| split (pre-merge) | rigid solve owns the frame, elastic block owns the modes |

Baseline is the corrected configuration of `shank_parity.md` sections 1-3: Young's modulus
`9e10`, damping ratio `0.02`, contact `kd = 0.35` on both legs, articulation relaxation `0.8`,
`block_sparse_joints`. `12.5 kg` payload dropped `5 cm`. The training preset is 2 substeps and
10 iterations.

```
python -m me.scripts.shank_parity.drop_elastic_solve
```

**Summary.**

1. Cost is the same for both arrangements at every point of the grid, within `2%`, and the
   current tree is `3%` slower than pre-merge at matched settings. Neither accounts for a
   noticeable training slowdown.
2. Substeps and iterations must be raised together. The minimum iteration count for stability
   rises with the substep count: 2 iterations suffice at 4 substeps, 10 are needed at 12 to 16,
   and 20 at 24 to 32. Refining the step alone destabilises the leg.
3. Substeps are the accuracy-limiting axis and are not converged at 16. Along the
   iteration-converged column the squash rises monotonically `13.39, 14.05, 14.52, 14.72 mm`
   with decrements `0.66, 0.48, 0.19`.
4. At the training preset the squash is `9%` below the `16x40` value, and iterations cannot
   close it: at 2 substeps, raising iterations from 10 to 40 changes the squash by `0.07%`.
5. The unified arrangement is less robust than the split one. Beyond the shared requirement of
   point 2 it also diverges at `8x2`, `12x5`, `16x20`, `32x20` and `32x40`, where split does
   not.
6. Return height does not converge over the range swept and is worse for the unified model 
   (compare to `shank_parity.md`).

---

## 1. Validation against `shank_parity.md`

The split arrangement at HEAD should reproduce the pre-merge measurements, which were taken on
the natively split tree. First-impact squash in mm:

| config | `shank_parity.md` | split, this study | difference |
| --- | ---: | ---: | ---: |
| 2x10 | 12.35 | 12.38 | 0.2% |
| 8x10 | 15.19 | 15.28 | 0.6% |
| 32x80 | 14.84 | 14.99 | 1.0% |
| 16x20 | 14.99 | 16.25 | 8.4% |

The first three agree. `16x20` does not, and that configuration is adjacent to the instability
of section 4. Values are not identical because the structure of the algorithm changed with the port, even for the split case.

Repeat runs of any configuration are bit-identical, so no measurement below is stochastic.

## 2. Convergence

First-impact squash in mm. Rows are substeps, columns are iterations.

unified

| substeps | 2 | 5 | 10 | 20 | 40 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 14.673 | 13.137 | 13.397 | 13.385 | 13.387 |
| 4 | 10.993 | 12.764 | 14.358 | 14.038 | 14.048 |
| 8 | diverged | 12.696 | 14.953 | 14.534 | 14.524 |
| 16 | diverged | diverged | 14.470 | 14.830 | 14.718 |

split

| substeps | 2 | 5 | 10 | 20 | 40 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.631 | 3.464 | 12.379 | 13.471 | 13.564 |
| 4 | 2.732 | 6.252 | 17.049 | 13.413 | 14.128 |
| 8 | 5.589 | 9.981 | 15.281 | 14.749 | 14.505 |
| 16 | diverged | diverged | 14.142 | 16.247 | 14.974 |

Entries at 2 and 5 iterations are either diverged or far from converged and carry no
information about accuracy. The unified `2x2` value of `14.673` happens to fall near the
reference, which is coincidence rather than convergence.

**Iterations.** Change in the unified squash on each doubling, as a percentage of the
40-iteration value:

| substeps | 10 to 20 | 20 to 40 |
| ---: | ---: | ---: |
| 2 | 0.09% | 0.01% |
| 4 | 2.28% | 0.07% |
| 8 | 2.89% | 0.07% |
| 16 | 2.45% | 0.76% |

Doubling from 10 moves the result by up to `2.9%`; doubling from 20 moves it by at most
`0.76%`. Twenty iterations is therefore sufficient for this model and ten is not.

**Substeps.** Read down the 40-iteration column, where the iteration axis is converged:
`13.387, 14.048, 14.524, 14.718`. This is monotone with decrements `0.661, 0.476, 0.194`. The
sequence has not converged at 16 substeps; the limit is plausibly `1` to `2%` above `14.718`.

The two axes move the squash in opposite directions.

**Split.** The split grid is noisier at low iteration counts, as expected from an arrangement
that requires more of them, and converges towards the same region: `14.505` at `8x40` and
`14.974` at `16x40` against unified's `14.524` and `14.718`.

## 3. Cost

Milliseconds per frame, measured on a separate instance after 10 warmup frames over 60 timed
frames, `wp.synchronize_device` around the timed region.

unified

| substeps | 2 | 5 | 10 | 20 | 40 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3.40 | 5.62 | 9.40 | 16.92 | 31.94 |
| 4 | 6.65 | 11.30 | 18.72 | 33.42 | 63.58 |
| 8 | 13.20 | 22.28 | 37.46 | 67.04 | 128.07 |
| 16 | 26.57 | 44.51 | 75.48 | 136.23 | 256.91 |

split

| substeps | 2 | 5 | 10 | 20 | 40 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3.30 | 5.60 | 9.43 | 17.20 | 32.03 |
| 4 | 6.63 | 11.13 | 19.12 | 34.15 | 64.39 |
| 8 | 13.17 | 22.54 | 38.05 | 68.46 | 129.36 |
| 16 | 26.48 | 44.95 | 75.90 | 137.29 | 260.30 |

The two arrangements differ by at most `2%` at any point, with no consistent sign. Cost scales linearly in
the product of substeps and iterations, as expected.

Against the pre-merge tree at matched settings:

| config | HEAD | pre-merge | ratio |
| ---: | ---: | ---: | ---: |
| 2x4 | 4.719 | 4.792 | 0.985 |
| 2x10 | 9.287 | 9.001 | 1.032 |
| 4x10 | 18.528 | 17.947 | 1.032 |
| 8x10 | 36.870 | 35.837 | 1.029 |

The current tree is approximately `3%` slower than pre-merge, flat across discretizations.

## 4. Stability

Maximum `|z|` of the elastic payload over `0.4 s`, in metres. The resting value is `0.5867`;
`div` marks a run that left the resting state or produced non-finite values.

unified

| substeps | 2 | 5 | 10 | 20 | 40 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.5867 | 0.5867 | 0.5867 | 0.5867 | 0.5867 |
| 8 | div | 0.5867 | 0.5867 | 0.5867 | 0.5867 |
| 12 | div | div | 0.5867 | 0.5867 | 0.5867 |
| 16 | div | div | 0.5867 | div | 0.5867 |
| 24 | div | div | div | 0.5867 | 0.5867 |
| 32 | div | div | div | div | div |

split

| substeps | 2 | 5 | 10 | 20 | 40 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.5867 | 0.5867 | 0.5867 | 0.5867 | 0.5867 |
| 8 | 0.6653 | 0.5867 | 0.5867 | 0.5867 | 0.5867 |
| 12 | div | 0.5867 | 0.5867 | 0.5867 | 0.5867 |
| 16 | div | div | 0.5867 | 0.5867 | 0.5867 |
| 24 | div | div | div | 0.5867 | 0.5867 |
| 32 | div | div | div | 0.5867 | 0.5867 |

**The dominant pattern is conventional: refining the step requires more iterations.** The
minimum stable iteration count rises with the substep count, roughly 2 at 4 substeps, 5 at 8
to 12, 10 at 16, and 20 at 24 to 32. Refining the step at a fixed iteration count moves a
configuration towards instability, not away from it. The discretization path used in
`shank_parity.md`, `2x10, 8x10, 16x20, 32x80`, raises both together and stays inside the
stable region.

**The unified arrangement is the less robust of the two.** It requires one more step on the
iteration axis at 8 and 12 substeps, and it fails at `32x20` and `32x40` where split does not.

**One pocket is genuinely non-monotone.** Unified is stable at `16x10`, diverges at `16x20`,
and is stable again at `16x40`. Adding iterations at a fixed step there moves a stable
configuration to an unstable one and back. This is the only place in the grid where more
iterations hurt, and it is not diagnosed.

The training preset `2x10` is far inside the stable region on both arrangements.

## 5. Return height

Unified, in mm.

| substeps | 2 | 5 | 10 | 20 | 40 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 26.15 | 25.29 | 26.87 | 26.76 | 26.75 |
| 4 | 32.67 | 32.33 | 35.60 | 34.80 | 34.75 |
| 8 | diverged | 38.58 | 40.38 | 40.62 | 40.50 |
| 16 | diverged | diverged | 33.23 | diverged | 43.73 |

The quantity rises with substeps through 8, falls at `16x10`, and rises again at `16x40`. It is
not converged over this range and no conclusion is drawn from it.

## 6. Control

Maximum absolute difference in PRDM squash between the two arrangements, over all 12
configurations: `0.000e+00`. The arrangement affects only the elastic path.

## Files

| path | what |
| --- | --- |
| `me/scripts/shank_parity/drop_elastic_solve.py` | discretization grid, cost and convergence |
| `me/data/shank_parity/drop_elastic_solve.json` | cached measurements |
| `newton/examples/robot/example_robot_compliant_shank_drop.py` | the drop model |
| `me/reports/shank_parity.md` | the chain/blade comparison this extends |
| `me/reports/elastic_unified_vs_split.md` | the two arrangements on isolated test cases |
