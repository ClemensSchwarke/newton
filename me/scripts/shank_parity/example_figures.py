"""Render the step example's tip response at the training preset and with the two static fixes.

The static fixes are the outcomes of report sections 1 and 2: the sparse articulation
relaxation raised off the pathological default, and the blade recalibrated to the modulus of
the Abaqus model the chain was fitted to. Nothing about the time step or the contact damping
is changed here.

The second figure overlays the training-preset traces so the two runs can be read against each
other, and both carry each model's own static equilibrium.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import newton.viewer
from newton.examples.robot import example_robot_compliant_shank_step as ex

FIGURES = "/tmp/claude-1000/newton-sparse-tests/me/figures/shank_parity"
TRAINING_RELAXATION = 0.65
TRAINING_YOUNG_MODULUS = 6.0e10
CORRECTED_RELAXATION = 0.8
CALIBRATED_YOUNG_MODULUS = 9.0e10
DURATION = 1.05
MODELS = (("prdm", "PRDM chain", "PRDM", "tab:blue"), ("elastic", "reduced elastic", "elastic", "tab:orange"))


class Args:
    centerline = None


def run(relaxation, modulus):
    ex.ARTICULATION_RELAXATION = relaxation
    ex.BLADE_YOUNG_MODULUS = modulus
    frames = int(DURATION * ex.FPS)
    example = ex.Example(newton.viewer.ViewerNull(num_frames=frames), Args)
    for _ in range(frames):
        example.step()
    example.print_summary()
    traces = {
        "t": np.asarray(example.trace["t"]),
        "reference": {key: np.asarray(example.reference[key]) for _, _, key, _ in MODELS},
        "true": np.asarray(example.true_reference),
    }
    for index, (key, _, _, _) in enumerate(MODELS):
        traces[key] = np.asarray(example.trace[key]) - example.rest_tip[index]
    return example, traces


print("===== training preset =====")
training_example, training = run(TRAINING_RELAXATION, TRAINING_YOUNG_MODULUS)
training_example.plot(f"{FIGURES}/step_training.png")

print(f"\n===== relaxation {CORRECTED_RELAXATION}, E = {CALIBRATED_YOUNG_MODULUS:.3g} =====")
fixed_example, fixed = run(CORRECTED_RELAXATION, CALIBRATED_YOUNG_MODULUS)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8.5, 6.5))
for axis, index, name in zip(axes, (1, 2), ("y", "z"), strict=True):
    for key, label, reference_key, color in MODELS:
        axis.plot(training["t"], training[key][:, index], color=color, lw=0.9, alpha=0.35, label=f"{label}, training")
        axis.plot(fixed["t"], fixed[key][:, index], color=color, lw=1.4, label=f"{label}, fixed")
        axis.axhline(
            fixed["reference"][reference_key][index - 1],
            color=color,
            ls="--",
            lw=1.4,
            label=f"{label} model equilibrium",
        )
    axis.axhline(fixed["true"][index - 1], color="black", ls="-.", lw=1.6, label="Abaqus true static")
    axis.set_ylabel(f"tip {name} deflection [m]")
    axis.grid(alpha=0.3)
axes[0].legend(fontsize=7, ncol=2)
axes[-1].set_xlabel("time [s]")
fig.suptitle(
    f"Tip response to a {ex.TIP_FORCE_Z:.0f} N step: relaxation {CORRECTED_RELAXATION} and "
    f"E = {CALIBRATED_YOUNG_MODULUS:.3g}, against the training preset"
)
fig.tight_layout()
output = f"{FIGURES}/step_static_fixes.png"
fig.savefig(output, dpi=140)
print(f"wrote {output}")
