"""Sweep the VBD discretization of the compliant-shank step example and record how close each
model gets to its own static solution.

The reference for each model is exact and solver-independent: the nonlinear equilibrium of the
rigid-link chain, and the retained modes' response to the same tip force. Anything short of
100% here is solver error.

Run at `rigid_articulation_relaxation = 0.8` so that these panels isolate step-size and
iteration error. At the training default of 0.65 the chain's answer is dominated by a separate
relaxation artifact -- see shank_parity/relaxation.py.
"""

import json

import numpy as np

import newton.viewer
from newton.examples.robot import example_robot_compliant_shank_step as ex

OUT_JSON = "/tmp/claude-1000/newton-sparse-tests/me/data/shank_parity/step_convergence.json"
DURATION = 1.0
SUBSTEP_SWEEP = [2, 4, 8, 16, 32, 64]
ITERATION_SWEEP = [10, 20, 40, 80, 160]
BASE_SUBSTEPS = 2
BASE_ITERATIONS = 10
BASE_RELAXATION = 0.8
CALIBRATED_YOUNG_MODULUS = 9.0e10
KEYS = {"prdm": "PRDM", "elastic": "elastic"}


class Args:
    centerline = None


def run(substeps, iterations, relaxation):
    ex.SUBSTEPS = substeps
    ex.BLADE_YOUNG_MODULUS = CALIBRATED_YOUNG_MODULUS
    ex.ITERATIONS = iterations
    ex.ARTICULATION_RELAXATION = relaxation
    frames = int((ex.STEP_TIME + DURATION) * ex.FPS)
    example = ex.Example(newton.viewer.ViewerNull(num_frames=frames), Args)
    for _ in range(frames):
        example.step()
    times = np.asarray(example.trace["t"])
    post = times >= ex.STEP_TIME
    out = {}
    for index, key in enumerate(("prdm", "elastic")):
        tip = (np.asarray(example.trace[key]) - example.rest_tip[index])[post]
        settled = tip[3 * tip.shape[0] // 4 :, 2]
        reference = float(example.reference[KEYS[key]][1])
        out[key] = {"disp_z": float(settled.mean()), "reference_z": reference}
        out[key]["fraction"] = out[key]["disp_z"] / reference
    return out


SWEEPS = [
    ("substeps", SUBSTEP_SWEEP, lambda v: (v, BASE_ITERATIONS, BASE_RELAXATION)),
    ("iterations", ITERATION_SWEEP, lambda v: (BASE_SUBSTEPS, v, BASE_RELAXATION)),
]

results = {}
for name, values, resolve in SWEEPS:
    results[name] = {}
    for value in values:
        substeps, iterations, relaxation = resolve(value)
        entry = run(substeps, iterations, relaxation)
        results[name][str(value)] = entry
        print(
            f"{name}={value:<6} sub={substeps:3d} it={iterations:3d} relax={relaxation:.2f}   "
            f"prdm={entry['prdm']['disp_z']:+.5f} ({entry['prdm']['fraction'] * 100:6.1f}%)   "
            f"elastic={entry['elastic']['disp_z']:+.5f} ({entry['elastic']['fraction'] * 100:6.1f}%)",
            flush=True,
        )

with open(OUT_JSON, "w") as handle:
    json.dump(results, handle, indent=2)

print(f"wrote {OUT_JSON}")
