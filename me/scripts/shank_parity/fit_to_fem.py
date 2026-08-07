"""Calibrate both shank models against the same Abaqus force sweep that produced the chain's
`stiffness = 6551`.

`compliance_advantures/code/prdm/fit_chain_deformation_to_fem.py` fits one scalar spring
stiffness to 100 random two-component tip loads on a 3D solid FEM blade, minimising the
sum-of-squared tip displacement error over both components. This script reproduces that fit
with the chain geometry the Newton examples use (which samples the cached centerline polyline
rather than the SVG path directly), and then runs the same objective on the reduced elastic
blade to fit its Young's modulus.

Reproducing 6551 validates that the chain statics used elsewhere in this report are the same
model the repo fitted. Fitting the blade the same way puts both models on one ground truth,
instead of matching the blade to the chain at a single load.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar, root

import newton
from newton.examples.robot import example_robot_compliant_shank_step as ex

FEM_CSV = "/home/clem/git/compliance_advantures/data/abaqus/batch_90GPa/force_sweep_results_3d_xcel_short.csv"
REPO_STIFFNESS = 6551.0
REPO_LINK_LENGTH = 0.0775263722
OUT_JSON = "/tmp/claude-1000/newton-sparse-tests/me/data/shank_parity/fit_to_fem.json"
OUT_PNG = "/tmp/claude-1000/newton-sparse-tests/me/figures/shank_parity/fit_to_fem.png"

centerline = ex.load_centerline(ex.resolve_centerline(None))
fem = np.loadtxt(FEM_CSV, delimiter=",", skiprows=1)
fem = fem[~np.any(np.isnan(fem), axis=1)]
forces, fem_disp = fem[:, :2], fem[:, 2:]


def chain_geometry(link_length=None):
    nodes = ex.prdm_nodes(centerline, ex.N_SEGMENTS)[:, 1:]
    segments = np.diff(nodes, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if link_length is not None:
        lengths = np.full_like(lengths, link_length)
    return lengths, np.arctan2(segments[:, 1], segments[:, 0])


def chain_tip(theta, lengths, rest):
    angles = rest + np.cumsum(theta)
    return np.array([np.sum(lengths * np.cos(angles)), np.sum(lengths * np.sin(angles))])


def chain_static(stiffness, force, lengths, rest):
    def residual(theta):
        angles = rest + np.cumsum(theta)
        dx = lengths * np.cos(angles)
        dy = lengths * np.sin(angles)
        arm_x = np.cumsum(dx[::-1])[::-1]
        arm_y = np.cumsum(dy[::-1])[::-1]
        return stiffness * theta - (force[1] * arm_x - force[0] * arm_y)

    solution = root(residual, np.zeros(lengths.size))
    return chain_tip(solution.x, lengths, rest) - chain_tip(np.zeros(lengths.size), lengths, rest)


def chain_sse(stiffness, lengths, rest):
    predicted = np.array([chain_static(stiffness, f, lengths, rest) for f in forces])
    return float(np.sum((predicted - fem_disp) ** 2))


def beam_compliance(mode_count=None):
    """2x2 tip compliance [m/N] mapping (fx, fy) in the centerline plane to (dx, dy)."""
    generator = newton.ModalGeneratorCurvedBeam(
        centerline=centerline,
        width=ex.BLADE_WIDTH,
        depth=ex.BLADE_DEPTH,
        plane="yz",
        mode_count=mode_count or 1,
        density=ex.BLADE_DENSITY,
        young_modulus=ex.BLADE_YOUNG_MODULUS,
        damping_ratio=ex.BLADE_DAMPING_RATIO,
    )
    stiffness, mass = generator._assemble()
    dof = stiffness.shape[0]
    tip = 6 * (generator.node_count - 1)
    free = np.setdiff1d(np.arange(dof), generator._fixed_dofs())
    if mode_count is not None:
        eigenvalues, modes = generator._solve_modes(stiffness, mass)
    columns = []
    for axis in (1, 2):
        force = np.zeros(dof)
        force[tip + axis] = 1.0
        if mode_count is None:
            displacement = np.zeros(dof)
            displacement[free] = np.linalg.solve(stiffness[np.ix_(free, free)], force[free])
        else:
            displacement = modes @ ((modes.T @ force) / eigenvalues)
        columns.append([displacement[tip + 1], displacement[tip + 2]])
    return np.array(columns).T


def fit_modulus(compliance):
    """Least-squares Young's modulus: displacement scales as 1/E, so the fit is linear."""
    predicted = forces @ compliance.T
    scale = float(np.sum(predicted * fem_disp) / np.sum(predicted * predicted))
    modulus = ex.BLADE_YOUNG_MODULUS / scale
    residual = scale * predicted - fem_disp
    return modulus, float(np.sqrt(np.sum(residual**2) / len(forces))), scale * predicted


results = {}
print(f"{len(forces)} FEM samples, |f| up to {np.max(np.linalg.norm(forces, axis=1)):.0f} N")

for label, link_length in (("example geometry", None), ("repo link length", REPO_LINK_LENGTH)):
    lengths, rest = chain_geometry(link_length)
    fit = minimize_scalar(chain_sse, bounds=(100.0, 20000.0), args=(lengths, rest), method="bounded")
    rms = float(np.sqrt(fit.fun / len(forces)))
    results[f"chain, {label}"] = {"stiffness": float(fit.x), "rms": rms, "link_length": float(lengths[0])}
    print(f"chain, {label:18s} link={lengths[0]:.7f}  k = {fit.x:8.1f}  RMS = {rms * 1e3:.3f} mm")

lengths, rest = chain_geometry()
chain_pred = np.array([chain_static(REPO_STIFFNESS, f, lengths, rest) for f in forces])
chain_rms = float(np.sqrt(np.sum((chain_pred - fem_disp) ** 2) / len(forces)))
results["chain at repo stiffness"] = {"stiffness": REPO_STIFFNESS, "rms": chain_rms}
print(f"chain at repo k={REPO_STIFFNESS:.0f}      RMS = {chain_rms * 1e3:.3f} mm")

blade_pred = {}
for label, mode_count in (("full beam FE", None), ("4 modes", 4), ("2 modes", 2)):
    compliance = beam_compliance(mode_count)
    modulus, rms, predicted = fit_modulus(compliance)
    blade_pred[label] = predicted
    results[f"blade, {label}"] = {"young_modulus": modulus, "rms": rms}
    print(f"blade, {label:14s}  E = {modulus:.4g} Pa  RMS = {rms * 1e3:.3f} mm")

compliance = beam_compliance(4)
predicted_at_training = forces @ compliance.T
training_rms = float(np.sqrt(np.sum((predicted_at_training - fem_disp) ** 2) / len(forces)))
results["blade at training modulus"] = {"young_modulus": ex.BLADE_YOUNG_MODULUS, "rms": training_rms}
print(f"blade at training E={ex.BLADE_YOUNG_MODULUS:.3g}  RMS = {training_rms * 1e3:.3f} mm")

magnitude = np.linalg.norm(forces, axis=1)
thresholds = [250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0]
band = {"threshold": [], "count": [], "chain_rms": [], "blade_rms": [], "blade_modulus": []}
for threshold in thresholds:
    mask = magnitude <= threshold
    subset_forces, subset_disp = forces[mask], fem_disp[mask]
    predicted = subset_forces @ compliance.T
    scale = float(np.sum(predicted * subset_disp) / np.sum(predicted * predicted))
    chain_subset = np.array([chain_static(REPO_STIFFNESS, f, lengths, rest) for f in subset_forces])
    band["threshold"].append(threshold)
    band["count"].append(int(mask.sum()))
    band["blade_modulus"].append(ex.BLADE_YOUNG_MODULUS / scale)
    band["blade_rms"].append(float(np.sqrt(np.sum((scale * predicted - subset_disp) ** 2) / mask.sum())))
    band["chain_rms"].append(float(np.sqrt(np.sum((chain_subset - subset_disp) ** 2) / mask.sum())))
    print(
        f"|f| <= {threshold:6.0f} N  n={mask.sum():3d}  blade E={band['blade_modulus'][-1]:9.4g}  "
        f"blade RMS={band['blade_rms'][-1] * 1e3:7.3f} mm   chain RMS={band['chain_rms'][-1] * 1e3:7.3f} mm"
    )
results["by_load_band"] = band

with open(OUT_JSON, "w") as handle:
    json.dump(results, handle, indent=2)

fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))
for axis, component, name in zip(axes[:2], (0, 1), ("x", "y"), strict=True):
    axis.scatter(
        fem_disp[:, component], chain_pred[:, component], s=12, alpha=0.6, label=f"chain, k={REPO_STIFFNESS:.0f}"
    )
    axis.scatter(
        fem_disp[:, component], blade_pred["4 modes"][:, component], s=12, alpha=0.6, label="blade, fitted E, 4 modes"
    )
    axis.scatter(
        fem_disp[:, component],
        predicted_at_training[:, component],
        s=12,
        alpha=0.6,
        label=f"blade, training E={ex.BLADE_YOUNG_MODULUS:.2g}",
    )
    limits = [fem_disp[:, component].min(), fem_disp[:, component].max()]
    axis.plot(limits, limits, "k--", lw=1.0)
    axis.set_xlabel(f"Abaqus tip {name} displacement [m]")
    axis.set_ylabel(f"model tip {name} displacement [m]")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
axes[2].plot(thresholds, np.array(band["chain_rms"]) * 1e3, "-o", label=f"chain, k={REPO_STIFFNESS:.0f}")
axes[2].plot(thresholds, np.array(band["blade_rms"]) * 1e3, "-o", label="blade, best-fit E per band")
axes[2].set_xlabel("tip load magnitude included [N]")
axes[2].set_ylabel("RMS tip displacement error [mm]")
axes[2].set_title("Where each model is valid")
axes[2].grid(alpha=0.3)
axes[2].legend(fontsize=8)

fig.suptitle("Both models against the Abaqus force sweep that calibrated the chain (xcel-short, 90 GPa)")
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140)
print(f"wrote {OUT_PNG}")
