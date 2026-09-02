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
4. The toggle costs the same either way, because both sides factor a block of the same
   width. The commit did not: on a 4096-world compliant ANYmal stepping an identical
   state it added `20.2 ms` of GPU time per frame, `18%`.
5. That cost is now recovered, so the unified arrangement is no longer the more expensive one
   and nothing is left to weigh against it. Leaving the reduced elastic bodies out of the
   articulation layout, so their anchored rows are never factored, and correcting the launch
   configuration of the per-elastic-body kernels takes the same measurement from `133.61` to
   `112.14 ms` against a pre-merge `111.99`.

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

These two numbers price the toggle, not the commit. `SolverVBD` sets

```python
self.elastic_body_block_width = int(model.elastic_max_mode_count) + 6
```

unconditionally, so both sides of the toggle allocate, assemble and factor a block of the same
width; `_elastic_frame_in_block` selects what fills it. The `n_m`-wide block exists only on the
pre-merge tree, which the toggle cannot reach. Both models here also carry one elastic body
with one mode, and at that size the step is launch-bound.

### 3.1 Cost of the commit

Compliant ANYmal, four blades of four modes each, 8 substeps, 10 iterations,
`block_sparse_joints`, `6b60fe9d` against `682e3e3b`.

The model has no policy, so it falls from its spawn height and folds up. The two trees track
each other until first impact and separate after it, which leaves them with different contact
sets and makes the contact kernels incomparable. The measurement below removes that: the
pre-merge tree dumps its state after 25 frames, both trees restore it before every timed
frame, and the contact count is identical by construction. The forward-kinematics checksum
agrees across trees to all printed digits, so the joint layout transplants exactly.

| worlds | contacts | pre-merge GPU | merged GPU | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 172032 | 37.17 ms | 43.90 ms | 1.181 |
| 4096 | 688128 | 111.99 ms | 132.21 ms | 1.181 |

Per kernel at 4096 worlds, milliseconds per frame:

| kernel | pre-merge | merged | delta |
| --- | ---: | ---: | ---: |
| `assemble_elastic_contacts` | 12.452 | 21.422 | +8.970 |
| `assemble_elastic_joints` | 2.443 | 6.538 | +4.095 |
| `accumulate_body_body_contacts_per_body` | 28.918 | 32.811 | +3.894 |
| `solve_elastic_body` -> `solve_elastic_body_tiled_10` | 1.787 | 5.305 | +3.518 |
| `update_duals_joint` | 0.930 | 1.551 | +0.621 |
| `solve_articulation_sparse_block32_scalar` | 37.832 | 37.461 | -0.371 |
| `assemble_articulation_joints_scalar` | 20.258 | 19.787 | -0.471 |
| total | 111.99 | 132.21 | +20.22 |

The articulation solve is unchanged because pinning is not removal: `_body_diagonal_contribution`
replaces the elastic body's 6x6 diagonal with `1e30 * I` and leaves its row in place, so
dimension, sparsity and block count are identical in both trees and dense factorization work
does not depend on the values.

The elastic path carries the whole difference, and it roughly doubles, `16.68 -> 33.27 ms`.
It is the smaller system per robot. The articulation holds 17 bodies at 6 DOF, so 102 DOF in
33 blocks of 36 floats; the elastic side holds 4 blades in `6 + n_m = 10` wide blocks, so 40
DOF in 400 floats. Those two counts overlap: the 4 blade frames are 24 of the 102 and 24 of the
40, which is precisely the pinned rows section 3.2 measures. The disparity is in how each is
parallelized, not in size: `solve_articulation_sparse_block32_scalar` gets 128 threads per
robot, while `assemble_elastic_joints` launches one scalar thread per elastic body and that
thread owns all 100 block entries. Growth tracks the width ratio `2.5` rather than
its cube, `x1.72` for `assemble_elastic_contacts`, `x2.68` for `assemble_elastic_joints`,
`x2.97` for the solve, so the cost is block traffic and not factorization flops.

### 3.2 Headroom in the articulation layout

Pinning leaves the elastic body in the layout, so the merged tree pays for rows whose delta it
then discards. Removing the blades from the model entirely shrinks the layout exactly as
excluding them would. 4096 worlds, milliseconds per frame:

| | bodies/robot | blocks/robot | joints/robot | solve | assemble | diagonal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| with blades | 17 | 33 | 21 | 27.844 | 20.214 | 2.792 |
| blades absent | 13 | 25 | 13 | 22.480 | 18.145 | 2.136 |

`8.09 ms` against a regression of `20.22 ms`, and that is an upper bound. Each blade contributes
two joints, its own free joint carrying the frame and modal coordinates, and the fixed
shank-to-blade joint. The free joints are removable outright because the elastic block owns
those degrees of freedom. The fixed joints are not: the shank is still constrained to the blade
root, so they would become one-sided constraints reading the frame from ``body_q``, cheaper to
assemble but not free. Claiming this requires `build_rigid_articulation_sparse_layout` to
exclude elastic bodies and re-link the joints reaching them.

Wall-clock ratios under this protocol are lower, `1.081` at 4096, because restoring four state
arrays each frame adds a host-to-device copy to both sides. GPU time is the comparable figure.

Free-running, without the state lock, the same trees give `90.34 -> 104.92 ms` wall at 4096,
`1.161`. That run is not contact-matched (688128 against 851968 and 1016365 for the locked and
free cases) and is reported only because it is the configuration closest to a training rollout.

This reproduces the compliance training regression: the run on `6b60fe9d` collects at `8.74 s`
per iteration against `9.95 s` on `3ef32904`, `+13.8%`, with `Perf/learning_time` unchanged.

### 3.3 Recovering the cost

Section 3.2 put the layout headroom at `8.09 ms`. Taking it, plus launch-configuration fixes on
the per-elastic-body kernels, returns the unified arrangement to the pre-merge cost. Same
state-locked protocol, 4096 worlds, `688128` contacts in every column:

| tree | GPU ms/frame |
| --- | ---: |
| pre-merge `6b60fe9d` | 111.99 |
| merged, as written | 133.61 |
| merged, after the changes below | 112.14 |

| kernel | before | after | delta |
| --- | ---: | ---: | ---: |
| `accumulate_body_body_contacts_per_body` | 32.45 | 22.75 | -9.70 |
| `solve_articulation_sparse_block32_scalar` | 37.55 | 28.88 | -8.67 |
| `assemble_articulation_joints_scalar` | 20.19 | 18.22 | -1.97 |
| `assemble_elastic_joints` | 6.55 | 5.12 | -1.43 |
| `assemble_articulation_body_diagonal_scalar` | 2.79 | 2.47 | -0.33 |
| `carry_articulation_excluded_body_q` | 0.00 | 0.30 | +0.30 |

Three changes produce this.

`build_rigid_articulation_sparse_layout` takes an `excluded_bodies` mask and leaves reduced
elastic bodies out of the layout, so their rows are never factored. The joints reaching them
are assembled against their live end by a parent-only branch alongside the child-only branch
that already served world-parented joints. A group made up entirely of excluded bodies is kept,
since the sparse path needs an articulation to run over. This is the `8.67 ms` on the solve and
`1.97` on the assembly, and it is exact rather than approximate: eliminating a body whose
diagonal is `1e30` contributes nothing to its neighbours, and the assembled shank row is
bit-identical either way.

`assemble_elastic_joints` launches one scalar thread per elastic body, which at the default
block size is about 16 thread blocks on a 170-SM GPU. It barely scaled with world count,
`5.91 ms` at 1024 worlds against `6.54` at 4096 for four times the work. `block_dim=32` spreads
the same warps across many more SMs and takes it to `2.31 ms` at 1024. Applied to the other
per-elastic-body kernels that write one row each; not applied to any kernel using
`wp.atomic_add`, where the launch shape would change float32 summation order.

The `9.70 ms` on `accumulate_body_body_contacts_per_body` is a side effect and was not designed.
Its body list is now the articulation bodies followed by the excluded ones, so the four
contact-heavy blades per robot are contiguous instead of scattered among thirteen rigid bodies
that carry almost no contacts. The thread groups in a block do comparable work again. The body
count is unchanged at seventeen per robot.

Two defects surfaced, both found by checking trajectories rather than timings. The body list for
contact accumulation was `layout.articulation_bodies`, so excluding the blades from the layout
also stopped their contacts being accumulated; that needs its own list. And
`_apply_sparse_delta_value_to_body` writes a pass-through pose for anchored bodies, which is the
only thing that populated ``state_out.body_q`` for them, so an excluded body had a stale pose
copied back over it every iteration and froze in place. `carry_articulation_excluded_body_q`
restores that write.

## Files

| path | what |
| --- | --- |
| `me/scripts/elastic_block_comparison.py` | accuracy, convergence and cost tables |
| `me/scripts/view_elastic_block_comparison.py` | free beam, both arrangements side by side |
| `me/scripts/view_elastic_contact_comparison.py` | sliding slab, both arrangements side by side |
| `me/math/elastic_sparse_regression.tex` | indefinite-block diagnosis preceding this work |
| `me/scripts/anymal_elastic_arrangement_cost.py` | free-running cost, one run per worktree |
| `me/scripts/anymal_elastic_state_locked_cost.py` | section 3.1, state-locked cost |

Both viewers advance each arrangement in its own model and copy the resulting poses into a
display model containing both, since a solver carries a single `_elastic_frame_in_block`
setting.
