"""Cost and convergence of the elastic leg in the drop test, unified against split.

Sweeps discretization for the reduced elastic leg under both arrangements of its degrees of
freedom: unified, where the floating frame and the modal coordinates are solved together in
one (6 + n_m) block, and split, where the rigid solve owns the frame and the elastic block
owns the modes. The PRDM chain is present in the same model and is unaffected by the
arrangement, so it doubles as a control: its numbers must match across the two.

Everything else is the corrected baseline of shank_parity.md sections 1-3.

Two questions are answered together. How much of a training slowdown the unified arrangement
accounts for, from the measured cost per step. And how few substeps and iterations the
elastic leg actually needs, from the distance to a converged reference, which shank_parity.md
established for the chain but not for the blade.

Results are cached in OUT_JSON keyed by arrangement and discretization, so re-running only
measures what is missing. Delete the file to force a full re-measurement.
"""

import json
import os
import time

import warp as wp

import newton.viewer
from newton.examples.robot import example_robot_compliant_shank_drop as ex

OUT_JSON = "me/data/shank_parity/drop_elastic_solve_c0.json"

SUBSTEP_AXIS = (2, 4, 8, 16)
ITERATION_AXIS = (2, 5, 10, 20, 40)
DISCRETIZATIONS = [(s, i) for s in SUBSTEP_AXIS for i in ITERATION_AXIS]
ARRANGEMENTS = (("unified", True), ("split", False))
REFERENCE = (16, 40)
REFERENCE_ARRANGEMENT = "unified"

RELAXATION = 0.8
CALIBRATED_YOUNG_MODULUS = 9.0e10
MATCHED_DAMPING_RATIO = 0.02
CONTACT_KD = 0.35
GRAVITY = 9.81
TIMED_FRAMES = 60
WARMUP_FRAMES = 10


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

    frames = int(ex.DURATION * ex.FPS)
    example = ex.Example(newton.viewer.ViewerNull(num_frames=frames), Args)
    example.solver._elastic_frame_in_block = unified

    for _ in range(frames):
        example.step()

    entry = {"device": wp.get_device().name, "substeps": substeps, "iterations": iterations}
    for row in example.summary():
        available = ex.PAYLOAD_MASS * GRAVITY * (ex.DROP_HEIGHT + row["compression"])
        entry[row["name"]] = {
            "squash": row["compression"] * 1e3,
            "rebound_height": row["rebound_height"] * 1e3,
            "peak_fraction": row["peak_energy"] / available,
        }

    timing_example = ex.Example(newton.viewer.ViewerNull(num_frames=frames), Args)
    timing_example.solver._elastic_frame_in_block = unified
    for _ in range(WARMUP_FRAMES):
        timing_example.step()
    wp.synchronize_device()
    start = time.perf_counter()
    for _ in range(TIMED_FRAMES):
        timing_example.step()
    wp.synchronize_device()
    elapsed = time.perf_counter() - start
    entry["ms_per_frame"] = 1.0e3 * elapsed / TIMED_FRAMES
    entry["ms_per_substep"] = entry["ms_per_frame"] / substeps
    return entry


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
            print(f"measured {key}", flush=True)

    reference = results[f"{REFERENCE_ARRANGEMENT}|{REFERENCE[0]}x{REFERENCE[1]}"]["elastic"]

    def grid(title, extract, fmt):
        print(f"\n{title}")
        for name, _ in ARRANGEMENTS:
            print(f"  {name}")
            print("    " + f"{'substeps':>9}" + "".join(f"{f'{it} iters':>12}" for it in ITERATION_AXIS))
            for substeps in SUBSTEP_AXIS:
                cells = ""
                for iterations in ITERATION_AXIS:
                    entry = results.get(f"{name}|{substeps}x{iterations}")
                    cells += fmt(extract(entry)) if entry else f"{'-':>12}"
                print(f"    {substeps:>9}" + cells)

    grid(
        "elastic leg, first-impact squash [mm]",
        lambda e: e["elastic"]["squash"],
        lambda v: f"{v:>12.3f}",
    )
    grid(
        f"elastic leg, squash deviation from {REFERENCE_ARRANGEMENT} "
        f"{REFERENCE[0]}x{REFERENCE[1]} [%]",
        lambda e: 100.0 * abs(e["elastic"]["squash"] - reference["squash"]) / abs(reference["squash"]),
        lambda v: f"{v:>12.2f}",
    )
    grid(
        "elastic leg, return height [mm]",
        lambda e: e["elastic"]["rebound_height"],
        lambda v: f"{v:>12.2f}",
    )
    grid("cost [ms/frame]", lambda e: e["ms_per_frame"], lambda v: f"{v:>12.2f}")

    print("\nPRDM chain control: max |squash difference| between arrangements [mm]")
    worst = 0.0
    for substeps, iterations in DISCRETIZATIONS:
        label = f"{substeps}x{iterations}"
        u = results[f"unified|{label}"]["PRDM"]["squash"]
        sp = results[f"split|{label}"]["PRDM"]["squash"]
        worst = max(worst, abs(u - sp))
    print(f"  {worst:.3e}")


if __name__ == "__main__":
    main()
