"""Solver-independent linear tip compliance of the two compliant-shank models.

The blade tip is loaded with a vertical force and the tip deflection computed three ways:

  full beam FE     -- every nodal DoF of the curved Euler-Bernoulli beam, linear
  modal truncated  -- the same beam projected on its lowest `m` clamped-root modes, linear
  PRDM chain       -- the N-segment torsional spring chain linearized about its rest shape

The reduced elastic body's deformation field is a linear modal expansion, so `modal truncated`
bounds what it can represent no matter how well the solver converges. Comparing it against the
full beam separates truncation error from the modulus error of report section 2, and the chain's
compliance against the same beam gives its stiffness offset in the linear limit.
"""

import json

import numpy as np

import newton
from newton.examples.robot import example_robot_compliant_shank_step as ex

OUT_JSON = "/tmp/claude-1000/newton-sparse-tests/me/data/shank_parity/static_reference.json"
MODE_COUNTS = [2, 4]
CALIBRATED_YOUNG_MODULUS = 9.0e10

ex.BLADE_YOUNG_MODULUS = CALIBRATED_YOUNG_MODULUS


def beam_tip_compliance(mode_count=None):
    """Tip deflection [m] per newton of vertical tip load, (y, z)."""
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
    force = np.zeros(dof)
    force[tip + 2] = 1.0
    if mode_count is None:
        free = np.setdiff1d(np.arange(dof), generator._fixed_dofs())
        displacement = np.zeros(dof)
        displacement[free] = np.linalg.solve(stiffness[np.ix_(free, free)], force[free])
    else:
        eigenvalues, modes = generator._solve_modes(stiffness, mass)
        displacement = modes @ ((modes.T @ force) / eigenvalues)
    return float(displacement[tip + 1]), float(displacement[tip + 2])


def chain_tip_compliance(stiffness):
    """Linearized tip deflection [m] per newton of vertical tip load, (y, z)."""
    nodes = ex.prdm_nodes(centerline, ex.N_SEGMENTS)[:, 1:]
    arm_y = nodes[-1, 0] - nodes[:-1, 0]
    arm_z = nodes[-1, 1] - nodes[:-1, 1]
    theta = arm_y / stiffness
    return float(-np.sum(theta * arm_z)), float(np.sum(theta * arm_y))


centerline = ex.load_centerline(ex.resolve_centerline(None))
beam_c = beam_tip_compliance()
modal_c = {m: beam_tip_compliance(m) for m in MODE_COUNTS}
chain_c = chain_tip_compliance(ex.SHANK_TARGET_KE)

print(f"linear tip compliance per N ({ex.BLADE_YOUNG_MODULUS:.3g} Pa beam, ke={ex.SHANK_TARGET_KE:.0f} chain):")
print(f"  full beam FE     dz={beam_c[1]:.3e} m/N")
for m in MODE_COUNTS:
    print(f"  modal, {m} modes   dz={modal_c[m][1]:.3e} m/N  ({modal_c[m][1] / beam_c[1] * 100:.1f}% of FE)")
print(
    f"  PRDM chain       dz={chain_c[1]:.3e} m/N  ({chain_c[1] / beam_c[1] * 100:.1f}% of FE, "
    f"{(beam_c[1] / chain_c[1] - 1.0) * 100:.1f}% stiffer)"
)

results = {
    "young_modulus": ex.BLADE_YOUNG_MODULUS,
    "shank_target_ke": ex.SHANK_TARGET_KE,
    "beam_compliance_per_n": {"dy": beam_c[0], "dz": beam_c[1]},
    "chain_compliance_per_n": {"dy": chain_c[0], "dz": chain_c[1]},
    "modal_compliance_per_n": {str(m): {"dy": modal_c[m][0], "dz": modal_c[m][1]} for m in MODE_COUNTS},
}
with open(OUT_JSON, "w") as handle:
    json.dump(results, handle, indent=2)
print(f"wrote {OUT_JSON}")
