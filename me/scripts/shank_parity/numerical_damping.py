"""Measure the solver's own numerical damping by running both shank models with zero
configured damping and fitting the resulting ring-down.

`SHANK_TARGET_KD = 0` and `BLADE_DAMPING_RATIO = 0` leave only the integrator's dissipation.
VBD's rigid drive and its modal update are both backward Euler, so the residual damping ratio
should follow `omega*dt/2`. The `of static` column flags where a model is not resolved enough
for its frequency and decay to mean anything.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import newton.viewer
from newton.examples.robot import example_robot_compliant_shank_step as ex

OUT_JSON = "/tmp/claude-1000/newton-sparse-tests/me/data/shank_parity/numerical_damping.json"
OUT_PNG = "/tmp/claude-1000/newton-sparse-tests/me/figures/shank_parity/numerical_damping.png"
PROBE_FORCE = 200.0
DURATION = 1.5
RELAXATION = 0.8
CALIBRATED_YOUNG_MODULUS = 9.0e10
SETTINGS = [(4, 20), (8, 40), (16, 40), (32, 80)]
TRAINING_SUBSTEPS = 2
COLORS = {"prdm": "tab:blue", "elastic": "tab:orange"}
NAMES = {"prdm": "PRDM chain", "elastic": "reduced elastic"}
KEYS = {"prdm": "PRDM", "elastic": "elastic"}


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


ex.ARTICULATION_RELAXATION = RELAXATION
ex.BLADE_YOUNG_MODULUS = CALIBRATED_YOUNG_MODULUS
ex.TIP_FORCE_Z = PROBE_FORCE
ex.SHANK_TARGET_KD = 0.0
ex.BLADE_DAMPING_RATIO = 0.0

results = {}
print(f"probe force {PROBE_FORCE:.0f} N, zero configured damping, relaxation {RELAXATION}")
print(
    f"{'sub':>5} {'it':>4} {'dt [ms]':>9} | {'model':>8} {'f [Hz]':>8} {'zeta_num':>9} {'w*dt/2':>8} {'of static':>10}"
)
for substeps, iterations in SETTINGS:
    ex.SUBSTEPS = substeps
    ex.ITERATIONS = iterations
    frames = int((ex.STEP_TIME + DURATION) * ex.FPS)
    example = ex.Example(newton.viewer.ViewerNull(num_frames=frames), Args)
    for _ in range(frames):
        example.step()
    times = np.asarray(example.trace["t"])
    mask = times >= ex.STEP_TIME
    dt = 1.0 / (ex.FPS * substeps)
    entry = {"substeps": substeps, "iterations": iterations, "dt": dt}
    for index, key in enumerate(("prdm", "elastic")):
        deflection = (np.asarray(example.trace[key]) - example.rest_tip[index])[:, 2]
        fit = fit_ringdown(times[mask], deflection[mask])
        fit["predicted_zeta"] = float(np.pi * fit["freq"] * dt)
        fit["fraction_of_static"] = fit["steady"] / float(example.reference[KEYS[key]][1])
        entry[key] = fit
        print(
            f"{substeps:>5} {iterations:>4} {dt * 1e3:>9.4f} | {key:>8} {fit['freq']:>8.2f} {fit['zeta']:>9.4f} "
            f"{fit['predicted_zeta']:>8.4f} {fit['fraction_of_static'] * 100:>9.1f}%"
        )
    results[f"{substeps}x{iterations}"] = entry

with open(OUT_JSON, "w") as handle:
    json.dump(results, handle, indent=2)

steps = [results[f"{s}x{i}"]["dt"] * 1e3 for s, i in SETTINGS]
fig, axis = plt.subplots(figsize=(7.5, 4.8))
for key in ("prdm", "elastic"):
    measured = [results[f"{s}x{i}"][key]["zeta"] for s, i in SETTINGS]
    predicted = [results[f"{s}x{i}"][key]["predicted_zeta"] for s, i in SETTINGS]
    axis.plot(steps, measured, "-o", color=COLORS[key], label=f"{NAMES[key]}, measured")
    axis.plot(steps, predicted, "--", color=COLORS[key], alpha=0.6, label=f"{NAMES[key]}, $\\omega \\Delta t/2$")
training_dt = 1.0 / (ex.FPS * TRAINING_SUBSTEPS) * 1e3
axis.axvline(training_dt, color="k", lw=1.0, alpha=0.7, label=f"training dt ({training_dt:.2f} ms)")
axis.axhline(0.01, color="gray", ls=":", lw=1.2, label="configured blade damping_ratio (0.01)")
axis.set_xscale("log")
axis.set_yscale("log")
axis.set_xlabel("substep dt [ms]")
axis.set_ylabel("numerical damping ratio")
axis.set_title("Solver dissipation with zero configured damping")
axis.grid(alpha=0.3, which="both")
axis.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140)
print(f"wrote {OUT_PNG}")
