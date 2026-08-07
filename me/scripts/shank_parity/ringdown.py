"""Measure frequency and damping of both compliant-shank models from the step ring-down.

Run at a converged discretization and at a load small enough to stay near-linear, so what is
measured is the model rather than the solver or the chain's geometric nonlinearity. A damped
sinusoid is fitted to the tip trace; peak-picking is unusable once the damping ratio passes a
few percent.

The chain's `damping` and the blade's `damping_ratio` are both absolute coefficients (relative
damping was removed in Newton 1.4.0: the VBD drive applies `drive_ke * err + drive_kd * vel`,
and the modal update solves `m v' = f - k q - c v`). A fixed absolute dashpot across a fixed
stiffness still yields a damping ratio that rises with frequency, `zeta = c*omega/(2k)`, while
the blade's ratio is constant by construction -- so the two only agree at one frequency.

The zero-damping run measures the solver's own numerical damping, which is subtracted to get
each model's physical damping; see shank_parity/numerical_damping.py for how that floor
scales with the step.

Each variant is expensive, so completed ones are cached in OUT_JSON and skipped on re-run.
Delete that file to force a full re-measurement.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import newton.viewer
from newton.examples.robot import example_robot_compliant_shank_step as ex

OUT_JSON = "/tmp/claude-1000/newton-sparse-tests/me/data/shank_parity/ringdown.json"
OUT_PNG = "/tmp/claude-1000/newton-sparse-tests/me/figures/shank_parity/ringdown.png"
SUBSTEPS = 32
ITERATIONS = 80
RELAXATION = 0.8
CALIBRATED_YOUNG_MODULUS = 9.0e10
DURATION = 1.5
PROBE_FORCE = 200.0
TRAINING_LABEL = "training (kd=2, ratio=0.01)"
VARIANTS = [
    ("undamped (kd=0, ratio=0)", 0.0, 0.0),
    ("half (kd=1, ratio=0.005)", 1.0, 0.005),
    (TRAINING_LABEL, 2.0, 0.01),
    ("double (kd=4, ratio=0.02)", 4.0, 0.02),
]


class Args:
    centerline = None


def damped_sinusoid(t, offset, amplitude, zeta, freq, phase):
    omega = 2.0 * np.pi * freq
    return offset + amplitude * np.exp(-zeta * omega * t) * np.cos(
        omega * np.sqrt(max(1.0 - zeta**2, 1.0e-9)) * t + phase
    )


def fit_ringdown(times, signal):
    times = times - times[0]
    offset = float(signal.mean())
    amplitude = 0.5 * float(signal.max() - signal.min())
    spectrum = np.abs(np.fft.rfft(signal - offset))
    frequency = float(np.fft.rfftfreq(signal.size, times[1] - times[0])[np.argmax(spectrum)])
    fitted, _ = curve_fit(damped_sinusoid, times, signal, p0=[offset, amplitude, 0.02, frequency, 0.0], maxfev=200000)
    return {"steady": float(fitted[0]), "freq": abs(float(fitted[3])), "zeta": abs(float(fitted[2]))}


ex.SUBSTEPS = SUBSTEPS
ex.BLADE_YOUNG_MODULUS = CALIBRATED_YOUNG_MODULUS
ex.ITERATIONS = ITERATIONS
ex.ARTICULATION_RELAXATION = RELAXATION
ex.TIP_FORCE_Z = PROBE_FORCE
frames = int((ex.STEP_TIME + DURATION) * ex.FPS)

results = {}
if os.path.exists(OUT_JSON):
    with open(OUT_JSON) as handle:
        results = json.load(handle)

for label, shank_kd, damping_ratio in VARIANTS:
    if label in results:
        print(f"cached: {label}", flush=True)
        continue
    ex.SHANK_TARGET_KD = shank_kd
    ex.BLADE_DAMPING_RATIO = damping_ratio
    example = ex.Example(newton.viewer.ViewerNull(num_frames=frames), Args)
    for _ in range(frames):
        example.step()
    times = np.asarray(example.trace["t"])
    mask = times >= ex.STEP_TIME
    entry = {"shank_kd": shank_kd, "damping_ratio": damping_ratio, "t": times.tolist()}
    for index, key in enumerate(("prdm", "elastic")):
        deflection = (np.asarray(example.trace[key]) - example.rest_tip[index])[:, 2]
        entry[key] = fit_ringdown(times[mask], deflection[mask])
        entry[f"{key}_trace"] = deflection.tolist()
    entry["reference"] = {k: float(example.reference[k][1]) for k in ("PRDM", "elastic")}
    entry["blade_modes_hz"] = np.asarray(example.frequencies).tolist()
    results[label] = entry
    with open(OUT_JSON, "w") as handle:
        json.dump(results, handle)
    print(f"measured: {label}", flush=True)

missing = [label for label, _, _ in VARIANTS if label not in results]
if missing:
    print(f"still missing {len(missing)} variant(s); re-run to continue: {missing}")
    raise SystemExit(0)

floor = {key: results[VARIANTS[0][0]][key]["zeta"] for key in ("prdm", "elastic")}
print(f"probe force {PROBE_FORCE:.0f} N, {SUBSTEPS} substeps, {ITERATIONS} iterations")
print(f"{'variant':>28} {'model':>8} {'f [Hz]':>8} {'zeta':>8} {'numerical':>10} {'physical':>9} {'predicted':>10}")
for label, shank_kd, damping_ratio in VARIANTS:
    entry = results[label]
    for key in ("prdm", "elastic"):
        measured = entry[key]["zeta"]
        physical = measured - floor[key]
        predicted = shank_kd / ex.SHANK_TARGET_KE * np.pi * entry[key]["freq"] if key == "prdm" else damping_ratio
        entry[key]["numerical_zeta"] = floor[key]
        entry[key]["physical_zeta"] = physical
        entry[key]["predicted_zeta"] = float(predicted)
        print(
            f"{label:>28} {key:>8} {entry[key]['freq']:>8.2f} {measured:>8.4f} {floor[key]:>10.4f} "
            f"{physical:>9.4f} {predicted:>10.4f}"
        )

training = results[TRAINING_LABEL]
modes = np.asarray(training["blade_modes_hz"])
nominal_kd = 2.0
nominal_ratio = 0.01
implied = nominal_kd / ex.SHANK_TARGET_KE * np.pi * modes
print("\ndamping ratio the chain's absolute kd produces at the blade's own mode frequencies:")
for frequency, zeta in zip(modes, implied, strict=True):
    print(f"  {frequency:8.2f} Hz -> zeta = {zeta:.4f}   (blade uses a constant {nominal_ratio})")

times = np.asarray(training["t"])
fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.5))
axes[0].plot(times, training["prdm_trace"], label="PRDM chain", lw=0.9)
axes[0].plot(times, training["elastic_trace"], label="reduced elastic", lw=0.9)
axes[0].axhline(training["reference"]["PRDM"], color="tab:blue", ls=":", lw=1.2, label="PRDM chain static")
axes[0].axhline(training["reference"]["elastic"], color="tab:orange", ls="--", lw=1.2, label="reduced elastic static")
axes[0].set_ylabel("tip z deflection [m]")
axes[0].set_xlabel("time [s]")
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=8)
axes[0].set_title(
    f"Ring-down after a {PROBE_FORCE:.0f} N tip step ({SUBSTEPS} substeps, {ITERATIONS} iterations)\n"
    f"chain {training['prdm']['freq']:.1f} Hz zeta {training['prdm']['zeta']:.3f}   "
    f"blade {training['elastic']['freq']:.1f} Hz zeta {training['elastic']['zeta']:.3f}"
)

axes[1].plot(modes, implied, "-o", label=f"chain, absolute kd={nominal_kd:g} N*m*s/rad")
axes[1].axhline(nominal_ratio, color="tab:orange", ls="--", label=f"blade, damping_ratio={nominal_ratio:g}")
axes[1].axhline(floor["elastic"], color="gray", ls=":", label=f"solver numerical damping ({floor['elastic']:.3f})")
axes[1].set_xlabel("blade mode frequency [Hz]")
axes[1].set_ylabel("modal damping ratio")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].grid(alpha=0.3, which="both")
axes[1].legend(fontsize=8)
axes[1].set_title("A fixed absolute dashpot damps high modes far harder than a fixed ratio")
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140)
print(f"\nwrote {OUT_PNG}")
