# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Unified Against Split Reduced Elastic Block
#
# The unified block solves a reduced elastic body's floating frame and its
# modal coordinates together in one (6 + n_m) dense system. The split
# arrangement is what the solver did before commit 06275246: the rigid solve
# owns the frame, the elastic block owns only the modes, and the two exchange
# information through a block Gauss-Seidel sweep. Both are reachable through
# SolverVBD._elastic_frame_in_block.
#
# Every basis here carries per-sample masses, so the coupling integrals are
# populated and the modes push back on their own frame. A basis without
# sample_mass leaves every coupling integral at zero, which removes the only
# thing the two arrangements disagree about and makes the comparison vacuous.
# ModalGeneratorBeam and a bare mode_shape_fn both take that uncoupled route;
# ModalGeneratorCurvedBeam, which builds the compliant blades, does not.
#
# Sample points come from the rendered surface mesh rather than a volume grid.
# A grid leaves the rendered and collided vertices without nearby samples, which
# both tears the surface visually and mis-weights the coupling integrals.
#
# Command:
#   python -m me.scripts.elastic_block_comparison
###########################################################################

import time

import numpy as np
import warp as wp

import newton
from newton.examples.basic._reduced_elastic import beam_render_sample_points, box_surface_mesh

DEVICE = "cuda:0"
SOLVE_PATHS = ("local", "block_sparse_joints")
CONFIGURATIONS = ((True, "unified"), (False, "split"))

BEAM_LENGTH = 1.0
BEAM_HALF_Y = 0.05
BEAM_HALF_Z = 0.04
BEAM_MASS = 1.0
BEAM_DEFLECTION = 0.12
BEAM_MODE_STIFFNESS = 10.0

RECOIL_STEPS = 480
RECOIL_DT = 1.0 / 480.0
RECOIL_ITERATIONS = 4

BLOCK_HALF = 0.05
BLOCK_MASS = 1.0
BLOCK_MODE_STIFFNESS = 200.0
BLOCK_DEFLECTION = 0.02
CARRIER_MASS = 2.0

CONVERGENCE_ITERATIONS = (1, 2, 4, 8, 16, 32, 64)
CONVERGENCE_REFERENCE_ITERATIONS = 800
CONVERGENCE_DT = 0.002

TIMING_ITERATIONS = 10
TIMING_WARMUP_STEPS = 20
TIMING_STEPS = 200
TIMING_DT = 0.002


def build_recoil_basis():
    """Free beam carrying one symmetric transverse bump mode, with sample masses.

    Symmetric on a uniform bar, so the mode has net linear coupling and no net angular
    coupling: releasing the bulge recoils the frame in pure translation.
    """
    points = beam_render_sample_points(
        BEAM_LENGTH,
        BEAM_HALF_Y,
        BEAM_HALF_Z,
        extra_points=((-0.5 * BEAM_LENGTH, 0.0, 0.0), (0.5 * BEAM_LENGTH, 0.0, 0.0)),
    )
    phi = np.zeros_like(points, dtype=np.float32)
    phi[:, 2] = 1.0 - (2.0 * points[:, 0] / BEAM_LENGTH) ** 2
    sample_mass = np.full(points.shape[0], BEAM_MASS / points.shape[0], dtype=np.float32)
    return newton.ModalBasis(
        sample_points=points,
        sample_phi=phi.reshape((-1, 1, 3)),
        sample_mass=sample_mass,
        mode_stiffness=[BEAM_MODE_STIFFNESS],
        mode_damping=[0.0],
        label="recoil_basis",
    )


def build_recoil_model(basis):
    builder = newton.ModelBuilder(gravity=0.0)
    shape_cfg = newton.ModelBuilder.ShapeConfig()
    shape_cfg.density = 0.0
    shape_cfg.has_shape_collision = False
    shape_cfg.has_particle_collision = False
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        com=wp.vec3(0.0, 0.0, 0.0),
        mass=BEAM_MASS,
        inertia=wp.mat33(0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02),
        mode_q=[BEAM_DEFLECTION],
        modal_basis=basis,
    )
    builder.add_shape_box(
        body,
        hx=0.5 * BEAM_LENGTH,
        hy=BEAM_HALF_Y,
        hz=BEAM_HALF_Z,
        cfg=shape_cfg,
    )
    builder.color()
    return builder.finalize(device=DEVICE)


def build_shear_basis():
    """Cube carrying one lateral shear mode that displaces the contacting face.

    phi_x runs from 1 at the bottom face to 0 at the top, so it is not a rigid translation
    and it moves exactly the surface that generates contacts. Sample masses give it net
    linear coupling to the frame as well.
    """
    points, _faces = box_surface_mesh(2.0 * BLOCK_HALF, BLOCK_HALF, BLOCK_HALF)
    phi = np.zeros_like(points, dtype=np.float32)
    phi[:, 0] = 0.5 - points[:, 2] / (2.0 * BLOCK_HALF)
    sample_mass = np.full(points.shape[0], BLOCK_MASS / points.shape[0], dtype=np.float32)
    return newton.ModalBasis(
        sample_points=points,
        sample_phi=phi.reshape((-1, 1, 3)),
        sample_mass=sample_mass,
        mode_stiffness=[BLOCK_MODE_STIFFNESS],
        mode_damping=[0.0],
        label="shear_basis",
    )


def build_coupled_contact_model(basis):
    """Rigid carrier jointed to a sheared cube sliding on frictional ground.

    Frame and modes couple twice over: inertially through the sample masses, and through
    contact, which acts on the deformed bottom face. This is the regime the unified block
    exists for.
    """
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.ke = 1.0e6
    cfg.kd = 0.0
    cfg.mu = 0.8
    cfg.margin = 0.0
    cfg.gap = 0.0

    builder = newton.ModelBuilder(gravity=-9.81, up_axis="Z")
    builder.add_ground_plane(cfg=cfg)
    carrier = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_identity()),
        mass=CARRIER_MASS,
        inertia=np.eye(3, dtype=np.float32) * 0.02,
    )
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.048), wp.quat_identity()),
        mass=BLOCK_MASS,
        inertia=np.eye(3, dtype=np.float32) * 0.01,
        mode_q=[BLOCK_DEFLECTION],
        modal_basis=basis,
    )
    builder.add_shape_box(body, hx=BLOCK_HALF, hy=BLOCK_HALF, hz=BLOCK_HALF, cfg=cfg)
    builder.add_joint_revolute(
        parent=carrier,
        child=body,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.202), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.body_qd[body] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    builder.color()
    return builder.finalize(device=DEVICE)


def _make_solver(model, articulation_solve, unified, iterations, **kwargs):
    solver = newton.solvers.SolverVBD(
        model, iterations=iterations, rigid_articulation_solve=articulation_solve, **kwargs
    )
    solver._elastic_frame_in_block = unified
    return solver


def _owner_q_start(model):
    owner_joint = int(model.elastic_joint.numpy()[0])
    return int(model.joint_q_start.numpy()[owner_joint])


def run_recoil_com_drift(articulation_solve, unified):
    """Centre-of-mass drift of a free vibrating beam, which must be exactly zero.

    With no external force the composite centre of mass x + (c/M) q is a conserved
    quantity, where c is the linear coupling integral. Any drift is the frame and the mode
    disagreeing about the momentum they exchange.
    """
    basis = build_recoil_basis()
    model = build_recoil_model(basis)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    q_start = _owner_q_start(model)

    coupling = np.array(basis.mode_coupling_linear[0], dtype=np.float64)
    com_factor = coupling / BEAM_MASS

    solver = _make_solver(model, articulation_solve, unified, iterations=RECOIL_ITERATIONS)

    def com():
        frame = state_0.body_q.numpy()[0, :3].astype(np.float64)
        modal = float(state_0.joint_q.numpy()[q_start + 7])
        return frame + com_factor * modal

    initial = com()
    worst = 0.0
    for _ in range(RECOIL_STEPS):
        solver.step(state_0, state_1, control, None, RECOIL_DT)
        state_0, state_1 = state_1, state_0
        worst = max(worst, float(np.linalg.norm(com() - initial)))
    return worst


def run_coupled(articulation_solve, unified, iterations, steps=1):
    basis = build_shear_basis()
    model = build_coupled_contact_model(basis)
    state_0, state_1 = model.state(), model.state()
    contacts, control = model.contacts(), model.control()
    solver = _make_solver(
        model,
        articulation_solve,
        unified,
        iterations=iterations,
        rigid_contact_k_start=1.0e6,
        elastic_contact_relaxation=1.0,
    )
    q_start = _owner_q_start(model)

    for _ in range(steps):
        model.collide(state_0, contacts)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, CONVERGENCE_DT)
        state_0, state_1 = state_1, state_0

    body_q = state_0.body_q.numpy().copy()
    modal = state_0.joint_q.numpy()[q_start + 7]
    return np.concatenate([body_q.reshape(-1), np.atleast_1d(modal)])


def time_steps(articulation_solve, unified):
    basis = build_shear_basis()
    model = build_coupled_contact_model(basis)
    state_0, state_1 = model.state(), model.state()
    contacts, control = model.contacts(), model.control()
    solver = _make_solver(
        model,
        articulation_solve,
        unified,
        iterations=TIMING_ITERATIONS,
        rigid_contact_k_start=1.0e6,
        elastic_contact_relaxation=1.0,
    )

    def advance(count):
        nonlocal state_0, state_1
        for _ in range(count):
            model.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, TIMING_DT)
            state_0, state_1 = state_1, state_0

    advance(TIMING_WARMUP_STEPS)
    wp.synchronize_device(DEVICE)
    start = time.perf_counter()
    advance(TIMING_STEPS)
    wp.synchronize_device(DEVICE)
    return 1.0e3 * (time.perf_counter() - start) / TIMING_STEPS


def report_coupling(basis, label):
    print(f"  {label:16s} linear={np.array(basis.mode_coupling_linear[0])} mode_mass={basis.mode_mass[0]:.5f}")


def report_accuracy():
    print("\n=== accuracy: centre-of-mass drift of a free vibrating beam [m] ===")
    print(f"exact value is zero; {RECOIL_STEPS} steps at dt={RECOIL_DT:.5f}, {RECOIL_ITERATIONS} iterations")
    print(f"{'path':22s} {'unified':>12s} {'split':>12s} {'split/unified':>14s}")
    for path in SOLVE_PATHS:
        values = {name: run_recoil_com_drift(path, unified) for unified, name in CONFIGURATIONS}
        ratio = values["split"] / max(values["unified"], 1.0e-30)
        print(f"{path:22s} {values['unified']:12.4e} {values['split']:12.4e} {ratio:14.2f}")


def report_convergence():
    print("\n=== convergence of one implicit step on the coupled contact model ===")
    print(f"distance to a shared unified {CONVERGENCE_REFERENCE_ITERATIONS}-iteration reference")
    for path in SOLVE_PATHS:
        reference = run_coupled(path, True, CONVERGENCE_REFERENCE_ITERATIONS)
        split_reference = run_coupled(path, False, CONVERGENCE_REFERENCE_ITERATIONS)
        gap = float(np.linalg.norm(split_reference - reference))
        print(f"\n  {path}   converged unified/split disagreement: {gap:.4e}")
        print(f"    {'iters':>6s} {'unified':>12s} {'split':>12s} {'split/unified':>14s}")
        for iterations in CONVERGENCE_ITERATIONS:
            errors = {}
            for unified, name in CONFIGURATIONS:
                state = run_coupled(path, unified, iterations)
                errors[name] = float(np.linalg.norm(state - reference))
            ratio = errors["split"] / max(errors["unified"], 1.0e-30)
            print(f"    {iterations:6d} {errors['unified']:12.4e} {errors['split']:12.4e} {ratio:14.2f}")


def report_speed():
    print("\n=== wall-clock per step (ms) ===")
    print(f"{'path':22s} {'unified':>12s} {'split':>12s} {'unified/split':>14s}")
    for path in SOLVE_PATHS:
        values = {name: time_steps(path, unified) for unified, name in CONFIGURATIONS}
        print(
            f"{path:22s} {values['unified']:12.4f} {values['split']:12.4f} {values['unified'] / values['split']:14.3f}"
        )


def main():
    wp.init()
    print(f"device: {DEVICE}, timing at {TIMING_ITERATIONS} iterations over {TIMING_STEPS} steps")
    print("\ncoupling integrals actually in use:")
    report_coupling(build_recoil_basis(), "recoil beam")
    report_coupling(build_shear_basis(), "shear cube")
    report_accuracy()
    report_convergence()
    report_speed()


if __name__ == "__main__":
    main()
