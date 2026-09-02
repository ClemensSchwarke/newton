"""Time response against substep count, one figure per arrangement and per example.

The tables in the drop reports reduce each run to two scalars. These keep the trajectory, so
the way a coarse step distorts the response is visible rather than inferred. Only the elastic
leg is plotted; the PRDM chain in the same models is unaffected by the arrangement.

The drop carries contact and the step load does not, which is the axis
`elastic_unified_vs_split.md` separates the two arrangements on. Squash and return height are
re-derived from each drop trace and printed against the cached sweep, so any drift from the
tree the tables were measured on is visible rather than silent.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

import newton.viewer
from newton.examples.robot import example_robot_compliant_shank_drop as ex
from newton.examples.robot import example_robot_compliant_shank_step as step_ex

SUBSTEP_AXIS = (2, 4, 8, 16)
ITERATIONS = 10
ARRANGEMENTS = (("unified", True), ("split", False))

RELAXATION = 0.8
CALIBRATED_YOUNG_MODULUS = 9.0e10
MATCHED_DAMPING_RATIO = 0.02
CONTACT_KD = 0.35

REFERENCE_JSON = "me/data/shank_parity/drop_elastic_solve_c0.json"
OUT_PNG = "me/figures/shank_parity/drop_elastic_time_response_{arrangement}.png"
STEP_OUT_PNG = "me/figures/shank_parity/step_elastic_time_response_{arrangement}.png"


class Args:
    centerline = None


def run(unified, substeps):
    ex.SUBSTEPS = substeps
    ex.ITERATIONS = ITERATIONS
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

    elastic = example.summary()[1]
    return {
        "t": np.asarray(example.trace["t"]),
        "drop": np.asarray(example.trace["elastic_z"]) - example.rest_carriage_z[1],
        "energy": np.asarray(example.trace["elastic_energy"]),
        "force": example.payload_force("elastic"),
        "squash": elastic["compression"] * 1e3,
        "rebound_height": elastic["rebound_height"] * 1e3,
        "peak_energy": elastic["peak_energy"],
        "peak_force": elastic["peak_force"],
    }


def draw(runs, arrangement, limits):
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(SUBSTEP_AXIS)))
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8.0, 9.0))
    for substeps, color in zip(SUBSTEP_AXIS, colors, strict=True):
        trace = runs[substeps]
        label = f"{substeps} substeps"
        axes[0].plot(trace["t"], trace["drop"], color=color, lw=1.4, label=label)
        axes[1].plot(trace["t"], trace["energy"], color=color, lw=1.4, label=label)
        axes[2].plot(trace["t"], trace["force"], color=color, lw=1.4, label=label)

    axes[0].axhline(-ex.DROP_HEIGHT, color="k", ls="--", lw=1.0, label="blade touches ground")
    axes[2].axhline(ex.PAYLOAD_MASS * 9.81, color="k", ls="--", lw=1.0, label="payload weight")
    axes[0].set_ylabel("payload drop [m]")
    axes[1].set_ylabel("energy stored in the leg [J]")
    axes[2].set_ylabel("force on the payload [N]")
    axes[2].set_xlabel("time [s]")
    for axis, key in zip(axes, ("drop", "energy", "force"), strict=True):
        axis.set_ylim(*limits[key])
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle(f"Reduced elastic leg, {arrangement}: drop response against substep count at {ITERATIONS} iterations")
    fig.tight_layout()
    path = OUT_PNG.format(arrangement=arrangement)
    fig.savefig(path, dpi=140)
    print(f"wrote {path}", flush=True)


def run_step(unified, substeps):
    step_ex.SUBSTEPS = substeps
    step_ex.ITERATIONS = ITERATIONS
    step_ex.ARTICULATION_RELAXATION = RELAXATION
    step_ex.BLADE_YOUNG_MODULUS = CALIBRATED_YOUNG_MODULUS
    step_ex.BLADE_DAMPING_RATIO = MATCHED_DAMPING_RATIO

    frames = int(step_ex.DURATION * step_ex.FPS)
    example = step_ex.Example(newton.viewer.ViewerNull(num_frames=frames), Args)
    example.solver._elastic_frame_in_block = unified
    for _ in range(frames):
        example.step()

    tip = np.asarray(example.trace["elastic"]) - example.rest_tip[1]
    row = next(r for r in example.summary() if r["name"] == "elastic")
    return {
        "t": np.asarray(example.trace["t"]),
        "y": tip[:, 1],
        "z": tip[:, 2],
        "equilibrium": np.asarray(example.reference["elastic"], dtype=float),
        "true": None if example.true_reference is None else np.asarray(example.true_reference, dtype=float),
        "disp": row["disp"],
        "freq": row["freq"],
        "zeta": row["zeta"],
        "peak_z": float(np.max(tip[:, 2])),
    }


def draw_step(runs, arrangement, limits):
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(SUBSTEP_AXIS)))
    first = runs[SUBSTEP_AXIS[0]]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8.0, 6.0))
    for axis, key, name, index in zip(axes, ("y", "z"), ("y", "z"), (0, 1), strict=True):
        for substeps, color in zip(SUBSTEP_AXIS, colors, strict=True):
            axis.plot(runs[substeps]["t"], runs[substeps][key], color=color, lw=1.4, label=f"{substeps} substeps")
        axis.axhline(
            first["equilibrium"][index], color="tab:orange", ls="--", lw=1.4, label="elastic model equilibrium"
        )
        if first["true"] is not None:
            axis.axhline(first["true"][index], color="black", ls="-.", lw=1.6, label="Abaqus true static")
        axis.set_ylim(*limits[key])
        axis.set_ylabel(f"tip {name} deflection [m]")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(
        f"Reduced elastic leg, {arrangement}: tip response to a {step_ex.TIP_FORCE_Z:.0f} N step "
        f"against substep count at {ITERATIONS} iterations"
    )
    fig.tight_layout()
    path = STEP_OUT_PNG.format(arrangement=arrangement)
    fig.savefig(path, dpi=140)
    print(f"wrote {path}", flush=True)


wp.init()
measured = {name: {s: run(unified, s) for s in SUBSTEP_AXIS} for name, unified in ARRANGEMENTS}

limits = {}
for key, pad in (("drop", 0.01), ("energy", 1.0), ("force", 120.0)):
    values = np.concatenate([measured[n][s][key] for n, _ in ARRANGEMENTS for s in SUBSTEP_AXIS])
    limits[key] = (float(values.min()) - pad, float(values.max()) + pad)

with open(REFERENCE_JSON) as handle:
    cached = json.load(handle)

print(
    f"\n{'cell':>16} {'squash now':>11} {'cached':>9} {'return now':>11} {'cached':>9} {'peak E':>8} {'peak F':>8} {'max rise':>9}"
)
for name, _unified in ARRANGEMENTS:
    for substeps in SUBSTEP_AXIS:
        trace = measured[name][substeps]
        entry = cached[f"{name}|{substeps}x{ITERATIONS}"]["elastic"]
        print(
            f"{name + '|' + str(substeps) + 'x' + str(ITERATIONS):>16} "
            f"{trace['squash']:>11.3f} {entry['squash']:>9.3f} "
            f"{trace['rebound_height']:>11.3f} {entry['rebound_height']:>9.3f} "
            f"{trace['peak_energy']:>8.2f} {trace['peak_force']:>8.1f} {1e3 * float(trace['drop'].max()):>9.1f}"
        )

for name, _unified in ARRANGEMENTS:
    draw(measured[name], name, limits)

step_measured = {name: {s: run_step(unified, s) for s in SUBSTEP_AXIS} for name, unified in ARRANGEMENTS}

step_limits = {}
for key in ("y", "z"):
    values = np.concatenate([step_measured[n][s][key] for n, _ in ARRANGEMENTS for s in SUBSTEP_AXIS])
    pad = 0.05 * (values.max() - values.min())
    step_limits[key] = (float(values.min()) - pad, float(values.max()) + pad)

print(f"\n{'cell':>16} {'settled z':>10} {'peak z':>9} {'f [Hz]':>8} {'zeta':>8}")
for name, _unified in ARRANGEMENTS:
    for substeps in SUBSTEP_AXIS:
        trace = step_measured[name][substeps]
        print(
            f"{name + '|' + str(substeps) + 'x' + str(ITERATIONS):>16} "
            f"{trace['disp']:>10.5f} {trace['peak_z']:>9.5f} {trace['freq']:>8.2f} {trace['zeta']:>8.4f}"
        )

for name, _unified in ARRANGEMENTS:
    draw_step(step_measured[name], name, step_limits)
