"""Stability of the reduced elastic leg in the drop test, unified against split.

Section 4 of drop_elastic_solve.md asks a different question from the rest of that study: not
how accurate a discretization is, but whether it survives at all. The payload starts at the
top of its drop and only ever falls, so a run whose peak height exceeds where it started has
gained energy it was never given, which is the solver diverging rather than the model
responding to an impact.

The grid extends further along the substep axis than the convergence study, because the
failure mode it looks for gets worse as the step is refined rather than better.

Results are cached in OUT_JSON keyed by arrangement and discretization, so re-running only
measures what is missing. Delete the file to force a full re-measurement.
"""

import json
import math
import os

import numpy as np
import warp as wp

import newton.viewer
from newton.examples.robot import example_robot_compliant_shank_drop as ex

OUT_JSON = "me/data/shank_parity/drop_elastic_stability.json"

SUBSTEP_AXIS = (4, 8, 12, 16, 24, 32)
ITERATION_AXIS = (2, 5, 10, 20, 40)
DISCRETIZATIONS = [(s, i) for s in SUBSTEP_AXIS for i in ITERATION_AXIS]
ARRANGEMENTS = (("unified", True), ("split", False))

REST_DURATION = 0.4

RELAXATION = 0.8
CALIBRATED_YOUNG_MODULUS = 9.0e10
MATCHED_DAMPING_RATIO = 0.02
CONTACT_KD = 0.35
DEPARTURE_LIMIT = 1.0


class Args:
    pass


def measure(unified, substeps, iterations):
    ex.SUBSTEPS = substeps
    ex.ITERATIONS = iterations
    ex.ARTICULATION_RELAXATION = RELAXATION
    ex.BLADE_YOUNG_MODULUS = CALIBRATED_YOUNG_MODULUS
    ex.BLADE_DAMPING_RATIO = MATCHED_DAMPING_RATIO
    ex.PRDM_CONTACT_KD = CONTACT_KD
    ex.BLADE_CONTACT_KD = CONTACT_KD
    ex.DURATION = REST_DURATION

    frames = int(ex.DURATION * ex.FPS)
    example = ex.Example(newton.viewer.ViewerNull(num_frames=frames), Args)
    example.solver._elastic_frame_in_block = unified
    rest_z = float(example.rest_carriage_z[1])

    for _ in range(frames):
        example.step()

    z = np.asarray(example.trace["elastic_z"])
    finite = bool(np.all(np.isfinite(z)))
    peak = float(np.max(np.abs(z))) if finite else math.inf
    departure = float(peak - rest_z) if finite else math.inf
    return {
        "device": wp.get_device().name,
        "substeps": substeps,
        "iterations": iterations,
        "rest_z": rest_z,
        "peak_abs_z": peak,
        "max_departure": departure,
        "stable": finite and departure < DEPARTURE_LIMIT,
    }


def main():
    wp.init()
    results = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as handle:
            results = json.load(handle)

    for name, unified in ARRANGEMENTS:
        for substeps, iterations in DISCRETIZATIONS:
            key = f"{name}|{substeps}x{iterations}"
            if key in results:
                continue
            results[key] = measure(unified, substeps, iterations)
            os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
            with open(OUT_JSON, "w") as handle:
                json.dump(results, handle, indent=2)
            print(f"measured {key}: {'ok' if results[key]['stable'] else 'div'}", flush=True)

    print(f"\nelastic leg over {REST_DURATION} s, peak |z| [m], start = "
          f"{results[f'unified|{SUBSTEP_AXIS[0]}x{ITERATION_AXIS[0]}']['rest_z']:.4f}")
    for name, _ in ARRANGEMENTS:
        print(f"  {name}")
        print("    " + f"{'substeps':>9}" + "".join(f"{f'{it} iters':>12}" for it in ITERATION_AXIS))
        for substeps in SUBSTEP_AXIS:
            cells = ""
            for iterations in ITERATION_AXIS:
                entry = results.get(f"{name}|{substeps}x{iterations}")
                if entry is None:
                    cells += f"{'-':>12}"
                elif entry["max_departure"] < DEPARTURE_LIMIT:
                    cells += f"{entry['peak_abs_z']:>12.4f}"
                else:
                    cells += f"{'div':>12}"
            print(f"    {substeps:>9}" + cells)


if __name__ == "__main__":
    main()
