# Convergence, accuracy and cost of the unified reduced elastic block against the split arrangement

A reduced elastic body's floating frame and its modal coordinates can be arranged two ways in
VBD. Commit `06275246` changed which one is used.

| arrangement | how the frame is solved |
| --- | --- |
| unified (current) | frame and modes together in one `(6 + n_m)` dense block |
| split (pre-merge) | rigid solve owns the frame, elastic block owns the modes, the two exchange information through a block Gauss-Seidel sweep |

Both are selectable at HEAD through the private `SolverVBD._elastic_frame_in_block`, so one
process can run the same model either way. Each result is additionally measured on the
pre-merge tree (`6b60fe9d`, in a `git worktree`) to establish that the toggle reproduces the
architecture it stands for.

Two test cases, chosen so that the frame and the modal coordinate exchange momentum:

| case | configuration |
| --- | --- |
| free beam | 1 m bar, 1 kg, zero gravity, one symmetric transverse bump mode deflected 0.12 m and released. No contact. Centre of mass is conserved analytically. |
| sliding slab | 0.24 x 0.24 x 0.04 m slab, 1 kg, pushed at 0.6 m/s across frictional ground (`ke = 2e4`, `kd = 30`, `mu = 0.5`), one lateral shear mode displacing the contacting face. Aspect ratio prevents tipping. |

```
python -m me.scripts.elastic_block_comparison
python -m me.scripts.view_elastic_block_comparison
python -m me.scripts.view_elastic_contact_comparison
```

**Summary.**

1. Both arrangements converge to the same solution, with and without contact.
2. Unified reaches its converged solution at 4 iterations without contact and 16 with it.
   Split requires approximately 24 and 64 respectively, the first figure after correcting an
   assembly-ordering lag described in section 1.1.
3. Below those thresholds the split result differs in sign, not only in magnitude: on the
   slab it terminates behind its starting position with the shear reversed.
4. Cost per step differs by less than 0.4% between the two.

---

## 1. Free beam: convergence without contact

Centre-of-mass drift over 480 frames, in metres. With no external force the quantity
`x + (c/M) q` is conserved exactly, so the drift measures inconsistency between the frame and
the modal coordinate in the momentum they exchange.

| iterations | unified | split | split, re-assembled |
| ---: | ---: | ---: | ---: |
| 4 | 4.1118e-04 | 3.9532e-02 | 1.8222e-02 |
| 8 | 4.1118e-04 | 1.8222e-02 | 5.7842e-03 |
| 16 | 4.1118e-04 | 5.7841e-03 | 8.1673e-04 |
| 24 | 4.1118e-04 | 2.1346e-03 | 2.7279e-04 |
| 48 | 4.1118e-04 | 2.7279e-04 | 2.7279e-04 |

The unified result is converged at 4 iterations and unchanged at 48. Both split results
decrease approximately as `1/iterations`.

At convergence the ordering reverses: split reaches `2.7e-04` against a unified floor of
`4.1e-04`. The difference between the arrangements is therefore in iteration count, not in
asymptotic accuracy.

The third column is explained in section 1.1. Pre-merge, running
`example_basic_reduced_elastic_frame_coupling` at its own `iterations=24`, gives `2.7279e-04`,
matching the re-assembled toggle at 24 iterations to five significant figures.

### 1.1 Ordering of the assembly relative to the frame solve

At HEAD the elastic block is assembled inside the per-colour loop, before `solve_rigid_body`,
and solved after it. In the split arrangement `solve_rigid_body` updates the elastic body's
own frame, so the modal solve uses a block assembled against the previous frame pose. The
pre-merge tree assembled and solved the modal system in one pass after all frames were
updated, so its modal solve used the current frame.

The two schemes differ as Jacobi does from Gauss-Seidel on the frame/modal pair:
`q_{k+1} = f(x_k)` against `q_{k+1} = f(x_{k+1})`. For a two-block splitting the Jacobi rate
is approximately the square of the Gauss-Seidel rate, that is about twice the iterations for
a given error.

Re-assembling the block after `solve_rigid_body` in the split arrangement, reported in the
third column, shifts the entire curve by exactly one doubling: the value at `N` iterations
becomes the value previously obtained at `2N`. The colour loop itself is not responsible; it
determines only where the assembly is placed.

## 2. Sliding slab: convergence under contact and friction

Final position and modal value after 240 frames. Friction displaces the contacting face
backwards relative to the body, so the converged solution has positive `x` and negative shear.

| iterations | unified x | unified mode | split x | split mode | re-assembled x | re-assembled mode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | +0.03529 | -0.03133 | -0.01554 | +0.02043 | -0.01578 | +0.02077 |
| 64 | +0.03595 | -0.03121 | +0.03582 | -0.03108 | +0.03593 | -0.03119 |
| 256 | +0.03595 | -0.03121 | +0.03612 | -0.03137 | +0.03613 | -0.03138 |

Relative to the unified 256-iteration solution, the unified result at 16 iterations deviates
by 1.8% in `x` and 0.4% in the modal value, and the split result by 143.2% and 165.5%, with
the opposite sign in both quantities. At 64 iterations the split deviations are 0.4% and 0.4%.

This case exercises friction in the tangent plane acting on a surface the mode displaces, so
frame and modes couple through the contact in addition to the inertial coupling.

Pre-merge gives `x = +0.04208`, `mode = -0.03220` at 16 iterations: the same sign and order of
magnitude as the converged solution.

### 2.1 The assembly ordering does not account for the contact case

The re-assembly of section 1.1 recovers pre-merge's efficiency exactly on the free beam. It
does not on the slab: at 16 iterations the re-assembled result remains sign-reversed
(`-0.01578`) where pre-merge gives `+0.04208`, and the gain at 64 iterations is marginal.

A second difference between pre-merge and the toggle therefore remains for contact
specifically, and is not identified. `accumulate_body_body_contacts_per_body` is not re-run by
the re-assembly, so `body_forces` stays stale across the frame solve; in the split arrangement
the modal rows are built by `assemble_elastic_contacts` from the deformed contact points
rather than from `body_forces`, so this is a candidate rather than an explanation.

## 3. Cost per step

10 iterations, 200 timed steps after 20 warmup steps, `cuda:0`, `wp.synchronize_device`
around the timed region.

| path | unified | split | unified/split |
| --- | ---: | ---: | ---: |
| local | 6.008 ms | 5.986 ms | 1.004 |
| block_sparse_joints | 4.331 ms | 4.339 ms | 0.998 |

Both arrangements factor a small dense block per elastic body. Six additional rows are not
measurable against the contact and joint assembly surrounding them.

## Files

| path | what |
| --- | --- |
| `me/scripts/elastic_block_comparison.py` | accuracy, convergence and cost tables |
| `me/scripts/view_elastic_block_comparison.py` | free beam, both arrangements side by side |
| `me/scripts/view_elastic_contact_comparison.py` | sliding slab, both arrangements side by side |
| `me/math/elastic_sparse_regression.tex` | indefinite-block diagnosis preceding this work |

Both viewers advance each arrangement in its own model and copy the resulting poses into a
display model containing both, since a solver carries a single `_elastic_frame_in_block`
setting.
