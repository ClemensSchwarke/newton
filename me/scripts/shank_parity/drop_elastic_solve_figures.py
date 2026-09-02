"""Squash and return height against substeps at fixed iterations, unified against split.

Reads the cached drop sweep and plots the fixed-iteration slice. Axis limits are fixed rather
than derived so the figure is stable if the sweep is re-measured. Measures nothing itself, so
it re-runs without a GPU.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUBSTEP_AXIS = (2, 4, 8, 16)
FIXED_ITERATIONS = 10
REFERENCE = (16, 40)
ARRANGEMENTS = (("unified", "unified", "tab:blue", "-o"), ("split", "split", "tab:orange", "--s"))
DROP_HEIGHT_MM = 50.0
PAYLOAD_MASS = 12.5
SQUASH_YLIM = (11.4, 21.8)
RETURN_YLIM = (-12.0, 82.0)

FIGURES = (
    (
        "me/data/shank_parity/drop_elastic_solve_c0.json",
        "me/figures/shank_parity/drop_elastic_solve_C0_substeps.png",
        "after the C0 snapshot fix",
    ),
)


def load(path):
    with open(path) as handle:
        return json.load(handle)


def series(results, arrangement, leg, metric):
    values = []
    for substeps in SUBSTEP_AXIS:
        entry = results.get(f"{arrangement}|{substeps}x{FIXED_ITERATIONS}")
        values.append(np.nan if entry is None else entry[leg][metric])
    return np.array(values, dtype=float)


def draw(results, out_png, subtitle):
    reference = results[f"unified|{REFERENCE[0]}x{REFERENCE[1]}"]["elastic"]
    positions = np.arange(len(SUBSTEP_AXIS))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for axis, metric, name in zip(
        axes, ("squash", "rebound_height"), ("first-impact compression", "return height"), strict=True
    ):
        for key, label, color, style in ARRANGEMENTS:
            axis.plot(positions, series(results, key, "elastic", metric), style, color=color, ms=6, label=label)
        axis.axhline(
            reference[metric],
            color="k",
            ls=":",
            lw=1.1,
            label=f"converged reference (unified {REFERENCE[0]}x{REFERENCE[1]})",
        )
        axis.set_xticks(positions)
        axis.set_xticklabels([str(s) for s in SUBSTEP_AXIS])
        axis.set_xlabel(f"substeps (iterations fixed at {FIXED_ITERATIONS})")
        axis.set_ylabel(f"{name} [mm]")
        axis.grid(alpha=0.3)

    axes[0].plot(
        positions,
        series(results, "unified", "PRDM", "squash"),
        ":^",
        color="0.55",
        ms=5,
        label="PRDM chain (control)",
    )
    axes[0].set_title("First-impact compression")
    axes[0].set_ylim(*SQUASH_YLIM)

    axes[1].axhline(DROP_HEIGHT_MM, color="k", ls="--", lw=1.0, label=f"lossless bounce ({DROP_HEIGHT_MM:.0f} mm)")
    axes[1].set_title("Return height")
    axes[1].set_ylim(*RETURN_YLIM)
    for key, _label, color, _style in ARRANGEMENTS:
        for position, value in zip(positions, series(results, key, "elastic", "rebound_height"), strict=True):
            if value > RETURN_YLIM[1]:
                axes[1].annotate(
                    f"{value:.0f}",
                    xy=(position, RETURN_YLIM[1]),
                    xytext=(position, RETURN_YLIM[1] - 9.0),
                    color=color,
                    ha="center",
                    fontsize=9,
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
                )

    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        f"Reduced elastic leg, {PAYLOAD_MASS:.1f} kg dropped {DROP_HEIGHT_MM:.0f} mm: "
        f"substep convergence at {FIXED_ITERATIONS} iterations, {subtitle}"
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}", flush=True)


for in_json, out_png, subtitle in FIGURES:
    draw(load(in_json), out_png, subtitle)
