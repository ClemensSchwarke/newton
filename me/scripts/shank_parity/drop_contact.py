"""Sweep contact damping against discretization for both shank models in the drop test.

Both legs are given the same contact `kd` in each run, so each model is seen under the
rigid-robot preset and under the value `elastic_shank.py` derives, across four
discretizations. Everything else is the corrected baseline of report sections 1-3.

Results are cached in OUT_JSON keyed by coefficient and discretization, so re-running only
measures what is missing. Delete the file to force a full re-measurement. The device is
recorded with each entry: the overdamped chain is sensitive enough at the finest step that
different GPUs disagree.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

import newton.viewer
from newton.examples.robot import example_robot_compliant_shank_drop as ex

OUT_JSON = "/tmp/claude-1000/newton-sparse-tests/me/data/shank_parity/drop_contact.json"
OUT_PNG = "/tmp/claude-1000/newton-sparse-tests/me/figures/shank_parity/drop_contact.png"

DISCRETIZATIONS = [(2, 10), (8, 10), (16, 20), (32, 80)]
CONTACT_KD = [1.0e5, 0.35]
RELAXATION = 0.8
CALIBRATED_YOUNG_MODULUS = 9.0e10
MATCHED_DAMPING_RATIO = 0.02
GRAVITY = 9.81
MODELS = (("PRDM", "PRDM chain", "tab:blue"), ("elastic", "reduced elastic", "tab:orange"))


class Args:
    centerline = None


def measure(contact_kd, substeps, iterations):
    ex.SUBSTEPS = substeps
    ex.ITERATIONS = iterations
    ex.ARTICULATION_RELAXATION = RELAXATION
    ex.BLADE_YOUNG_MODULUS = CALIBRATED_YOUNG_MODULUS
    ex.BLADE_DAMPING_RATIO = MATCHED_DAMPING_RATIO
    ex.PRDM_CONTACT_KD = contact_kd
    ex.BLADE_CONTACT_KD = contact_kd
    frames = int(ex.DURATION * ex.FPS)
    example = ex.Example(newton.viewer.ViewerNull(num_frames=frames), Args)
    for _ in range(frames):
        example.step()
    entry = {"device": wp.get_device().name}
    for row in example.summary():
        available = ex.PAYLOAD_MASS * GRAVITY * (ex.DROP_HEIGHT + row["compression"])
        entry[row["name"]] = {
            "squash": row["compression"] * 1e3,
            "rebound_height": row["rebound_height"] * 1e3,
            "peak_energy": row["peak_energy"],
            "energy_available": available,
            "peak_fraction": row["peak_energy"] / available,
        }
    return entry


results = {}
if os.path.exists(OUT_JSON):
    with open(OUT_JSON) as handle:
        results = json.load(handle)

for contact_kd in CONTACT_KD:
    for substeps, iterations in DISCRETIZATIONS:
        key = f"kd={contact_kd:g}|{substeps}x{iterations}"
        if key in results:
            continue
        results[key] = measure(contact_kd, substeps, iterations)
        with open(OUT_JSON, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"measured {key}", flush=True)


labels = [f"{s}x{i}" for s, i in DISCRETIZATIONS]
print(f"\n{'config':>8} " + " ".join(f"{m:>26}" for _, m, _ in MODELS))
for contact_kd in CONTACT_KD:
    print(f"contact kd = {contact_kd:g}      squash / return height [mm]")
    for label in labels:
        entry = results[f"kd={contact_kd:g}|{label}"]
        cells = " ".join(f"{entry[k]['squash']:>12.2f} {entry[k]['rebound_height']:>13.2f}" for k, _, _ in MODELS)
        print(f"{label:>8} {cells}")

fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
positions = np.arange(len(labels))
for axis, metric, name in zip(
    axes, ("squash", "rebound_height"), ("first-impact compression", "return height"), strict=True
):
    for key, model_label, color in MODELS:
        for contact_kd, style in zip(CONTACT_KD, ("--o", "-o"), strict=True):
            entries = [results[f"kd={contact_kd:g}|{lab}"][key] for lab in labels]
            values = [e[metric] for e in entries]
            axis.plot(positions, values, style, color=color, ms=5, label=f"{model_label}, kd={contact_kd:g}")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_xlabel("substeps x iterations")
    axis.set_ylabel(f"{name} [mm]")
    axis.grid(alpha=0.3)
axes[0].axhline(0.0, color="k", lw=0.9, label="no compression (rigid leg)")
axes[0].set_title("First-impact compression")
axes[1].axhline(ex.DROP_HEIGHT * 1e3, color="k", ls="--", lw=1.0, label="lossless bounce")
axes[1].set_title(f"Return height (lossless = {ex.DROP_HEIGHT * 1e3:.0f} mm)")
axes[0].legend(fontsize=8)
axes[1].legend(fontsize=8)
fig.suptitle(
    f"Compliant shank drop, {ex.PAYLOAD_MASS:.1f} kg from {ex.DROP_HEIGHT * 100:.0f} cm: "
    "contact damping against discretization"
)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140)
print(f"\nwrote {OUT_PNG}")
