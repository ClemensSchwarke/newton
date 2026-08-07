"""Analytic fundamental of each model against the frequency measured from the ring-down.

The blade's analytic value is the first eigenvalue of the beam FE its modal basis is built
from. The chain has no such value in the repo, so it is assembled here from the same link
geometry the example builds: 10 planar links with the box inertias of
`_build_prdm_shank`, torsional springs of `SHANK_TARGET_KE` on all ten joints, base pivot
fixed, gravity off. Generalized coordinates are the relative joint angles, in which the
stiffness matrix is `ke * I`.

The chain is a mechanism, so its frequency depends on the configuration it is linearized about.
Both the undeflected pose and the 200 N equilibrium the ring-down probes are reported.

Measured values are read from the ring-down cache, not re-simulated.
"""

import json

import numpy as np
from scipy.optimize import root

import newton
from newton.examples.robot import example_robot_compliant_shank_step as ex

RINGDOWN_JSON = "/tmp/claude-1000/newton-sparse-tests/me/data/shank_parity/ringdown.json"
OUT_JSON = "/tmp/claude-1000/newton-sparse-tests/me/data/shank_parity/frequencies.json"
PROBE_FORCE = 200.0
CALIBRATED_YOUNG_MODULUS = 9.0e10


def chain_geometry():
    nodes = ex.prdm_nodes(ex.load_centerline(ex.resolve_centerline(None)), ex.N_SEGMENTS)[:, 1:]
    segments = np.diff(nodes, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    return nodes, lengths, np.arctan2(segments[:, 1], segments[:, 0])


def chain_nodes_at(angles, lengths, origin):
    nodes = [origin]
    for length, angle in zip(lengths, angles, strict=True):
        nodes.append(nodes[-1] + length * np.array([np.cos(angle), np.sin(angle)]))
    return np.array(nodes)


def chain_frequency(angles, lengths, origin):
    """Fundamental [Hz] of the planar chain linearized about `angles`, in relative coordinates."""
    nodes = chain_nodes_at(angles, lengths, origin)
    mass = ex.BLADE_DENSITY * ex.BLADE_WIDTH * ex.BLADE_DEPTH * lengths
    inertia = mass / 12.0 * (ex.BLADE_DEPTH**2 + lengths**2)
    n = lengths.size
    generalized_mass = np.zeros((n, n))
    for i in range(n):
        com = 0.5 * (nodes[i] + nodes[i + 1])
        jacobian = np.zeros((2, n))
        for j in range(i + 1):
            arm = com - nodes[j]
            jacobian[:, j] = (-arm[1], arm[0])
        angular = np.zeros(n)
        angular[: i + 1] = 1.0
        generalized_mass += mass[i] * jacobian.T @ jacobian + inertia[i] * np.outer(angular, angular)
    stiffness = ex.SHANK_TARGET_KE * np.eye(n)
    eigenvalues = np.linalg.eigvals(np.linalg.solve(generalized_mass, stiffness))
    return float(np.sqrt(np.sort(eigenvalues.real)[0]) / (2.0 * np.pi))


def chain_equilibrium(lengths, rest, force):
    def residual(theta):
        angles = rest + np.cumsum(theta)
        arm_x = np.cumsum((lengths * np.cos(angles))[::-1])[::-1]
        arm_y = np.cumsum((lengths * np.sin(angles))[::-1])[::-1]
        return ex.SHANK_TARGET_KE * theta - (force[1] * arm_x - force[0] * arm_y)

    return root(residual, np.zeros(lengths.size)).x


def blade_frequencies(modulus):
    generator = newton.ModalGeneratorCurvedBeam(
        centerline=ex.load_centerline(ex.resolve_centerline(None)),
        width=ex.BLADE_WIDTH,
        depth=ex.BLADE_DEPTH,
        plane="yz",
        mode_count=ex.BLADE_MODE_COUNT,
        density=ex.BLADE_DENSITY,
        young_modulus=modulus,
        damping_ratio=ex.BLADE_DAMPING_RATIO,
    )
    stiffness, mass = generator._assemble()
    eigenvalues, _ = generator._solve_modes(stiffness, mass)
    return np.sqrt(eigenvalues) / (2.0 * np.pi)


nodes, lengths, rest = chain_geometry()
origin = nodes[0]
chain_rest = chain_frequency(rest, lengths, origin)
theta = chain_equilibrium(lengths, rest, np.array([0.0, PROBE_FORCE]))
chain_loaded = chain_frequency(rest + np.cumsum(theta), lengths, origin)
blade = blade_frequencies(CALIBRATED_YOUNG_MODULUS)

with open(RINGDOWN_JSON) as handle:
    ringdown = json.load(handle)
undamped = ringdown["undamped (kd=0, ratio=0)"]
measured = {"chain": undamped["prdm"]["freq"], "blade": undamped["elastic"]["freq"]}

results = {
    "chain": {"analytic_rest": chain_rest, "analytic_loaded": chain_loaded, "measured": measured["chain"]},
    "blade": {"analytic": float(blade[0]), "measured": measured["blade"], "modes": blade.tolist()},
    "probe_force": PROBE_FORCE,
}
with open(OUT_JSON, "w") as handle:
    json.dump(results, handle, indent=2)

print(f"chain analytic, undeflected      {chain_rest:8.3f} Hz")
print(f"chain analytic, {PROBE_FORCE:.0f} N equilibrium {chain_loaded:8.3f} Hz")
print(
    f"chain measured                   {measured['chain']:8.3f} Hz   "
    f"({(measured['chain'] / chain_loaded - 1.0) * 100:+.2f}% vs loaded)"
)
print(f"blade analytic (mode 1)          {blade[0]:8.3f} Hz")
print(
    f"blade measured                   {measured['blade']:8.3f} Hz   "
    f"({(measured['blade'] / blade[0] - 1.0) * 100:+.2f}%)"
)
print(f"blade analytic modes             {np.round(blade, 2).tolist()}")
print(f"wrote {OUT_JSON}")
