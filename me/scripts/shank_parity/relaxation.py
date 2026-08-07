"""Map the PRDM chain's converged tip deflection against rigid_articulation_relaxation, for
several iteration counts.

The block-sparse articulation solve scales its Newton update by `dx = dx_in * relaxation`. At
a fixed, low iteration count the iteration is stopped long before that scaling washes out, so
the fixed point it settles on -- and it is a genuine fixed point, stable for seconds -- depends
on the relaxation factor. This sweep shows how strongly, and how many iterations are needed
before the answer stops depending on it.

Every point is a settled value, not a transient: the chain reaches it within ~0.25 s of the
step and holds it.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import newton.viewer
from newton.examples.robot import example_robot_compliant_shank_step as ex

OUT_JSON = "/tmp/claude-1000/newton-sparse-tests/me/data/shank_parity/relaxation.json"
OUT_PNG = "/tmp/claude-1000/newton-sparse-tests/me/figures/shank_parity/relaxation.png"
SUBSTEPS = 2
DURATION = 0.8
RELAXATION_SWEEP = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.45,
    0.5,
    0.52,
    0.55,
    0.6,
    0.625,
    0.65,
    0.7,
    0.74,
    0.76,
    0.78,
    0.8,
    0.82,
    0.84,
    0.86,
    0.88,
    0.9,
    0.94,
    0.98,
    1.0,
]
ITERATION_SWEEP = [10, 20, 40, 80]
TRAINING_RELAXATION = 0.65


class Args:
    centerline = None


def settled_fraction(relaxation, iterations):
    ex.SUBSTEPS = SUBSTEPS
    ex.ITERATIONS = iterations
    ex.ARTICULATION_RELAXATION = relaxation
    frames = int((ex.STEP_TIME + DURATION) * ex.FPS)
    example = ex.Example(newton.viewer.ViewerNull(num_frames=frames), Args)
    for _ in range(frames):
        example.step()
    times = np.asarray(example.trace["t"])
    deflection = (np.asarray(example.trace["prdm"]) - example.rest_tip[0])[:, 2][times >= ex.STEP_TIME]
    settled = deflection[3 * deflection.size // 4 :]
    return float(settled.mean() / float(example.reference["PRDM"][1]))


results = {}
if os.path.exists(OUT_JSON):
    with open(OUT_JSON) as handle:
        results = json.load(handle)

for iterations in ITERATION_SWEEP:
    results.setdefault(str(iterations), {})
    for relaxation in RELAXATION_SWEEP:
        if str(relaxation) in results[str(iterations)]:
            continue
        fraction = settled_fraction(relaxation, iterations)
        results[str(iterations)][str(relaxation)] = fraction
        print(f"iterations={iterations:4d} relaxation={relaxation:5.3f}  {fraction * 100:10.1f}% of static", flush=True)
        with open(OUT_JSON, "w") as handle:
            json.dump(results, handle, indent=2)

fig, axis = plt.subplots(figsize=(8.0, 5.0))
for iterations in ITERATION_SWEEP:
    values = [results[str(iterations)][str(r)] * 100.0 for r in RELAXATION_SWEEP]
    axis.plot(RELAXATION_SWEEP, values, "-o", ms=4, label=f"{iterations} iterations")
axis.axhline(100.0, color="k", ls="--", lw=1.0, label="own static solution")
axis.axvline(TRAINING_RELAXATION, color="tab:red", lw=1.0, alpha=0.7, label=f"training default ({TRAINING_RELAXATION})")
axis.set_ylim(-500.0, 500.0)
axis.set_xlabel("rigid_articulation_relaxation")
axis.set_ylabel("settled tip z deflection [% of static]")
axis.grid(alpha=0.3)
axis.legend(fontsize=8)
fig.suptitle(f"PRDM chain, block_sparse_joints, {SUBSTEPS} substeps: the answer depends on the relaxation factor")
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140)
print(f"wrote {OUT_PNG}")
