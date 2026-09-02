# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cost of the reduced elastic solve on the compliant ANYmal, scaled in world count.

Run once per git worktree to compare trees. The pre-merge tree carries a modal-only
``n_m`` block; from ``06275246`` the block is ``6 + n_m`` wide in both arrangements, so the
private ``_elastic_frame_in_block`` toggle does not recover the pre-merge cost and only the
tree comparison is meaningful.
"""

import collections
import json
import time

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

import newton.examples.robot.example_robot_anymal_compliant_elastic as ex

USD_PATH = "/home/clem/git/isaac/compliance/source/compliance/compliance/assets/data/Anymal-D-Compliant/anymal_d_compliant_no_shanks.usd"
CENTERLINE = "/home/clem/git/isaac/compliance/source/compliance/compliance/assets/data/Anymal-D-Compliant/xcel_short_exported_centerline.npy"

WORLD_COUNTS = (64, 256, 1024, 4096)
SUBSTEPS = 8
ITERATIONS = 10
FRAME_DT = 1.0 / 50.0
CONTACT_KE = 8.0e5
JOINT_KE = 1.0e6
ARTICULATION_SOLVE = "block_sparse_joints"
ARTICULATION_RELAXATION = 0.8

WARMUP_FRAMES = 5
TIMED_FRAMES = 20
DEVICE = "cuda:0"

TAG = "tree"
OUT_JSON = "me/data/anymal_elastic_cost_scaling.json"


def build_model(world_count):
    centerline = np.load(CENTERLINE)
    shank_pose = ex._shank_world_poses(USD_PATH)
    ab = ex._setup_articulation(USD_PATH)

    shank_index = {}
    for body_idx, label in enumerate(ab.body_label):
        short = label.rsplit("/", 1)[-1]
        for leg, _ in ex.LEGS:
            if short == f"{leg}_SHANK":
                shank_index[leg] = body_idx

    shape_cfg = newton.ModelBuilder.ShapeConfig()
    shape_cfg.ke = CONTACT_KE
    shape_cfg.kd = ex.CONTACT_KD
    shape_cfg.mu = ex.CONTACT_MU

    robot_shapes = list(range(ab.shape_count))
    for leg, is_left in ex.LEGS:
        nodes = ex._shank_local_nodes(centerline, is_left)
        generator = newton.ModalGeneratorCurvedBeam(
            centerline=nodes,
            width=ex.BLADE_WIDTH,
            depth=ex.BLADE_DEPTH,
            mode_count=ex.BLADE_MODE_COUNT,
            density=ex.BLADE_DENSITY,
            young_modulus=ex.BLADE_YOUNG_MODULUS,
            damping_ratio=ex.BLADE_DAMPING_RATIO,
            label=f"{leg}_blade_basis",
        )
        basis = generator.build()
        blade = ab.add_body_elastic(
            xform=shank_pose[leg],
            mass=generator.total_mass,
            com=wp.vec3(*generator.center_of_mass),
            inertia=wp.mat33(*generator.inertia.flatten()),
            modal_basis=basis,
            lock_inertia=True,
            label=f"{leg}_blade",
        )
        root_local = wp.transform(wp.vec3(*generator.root_local), wp.quat_identity())
        ab.add_joint_fixed(
            parent=shank_index[leg],
            child=blade,
            parent_xform=root_local,
            child_xform=root_local,
            label=f"{leg}_shank_to_blade",
        )
        vertices, indices = generator.surface_mesh()
        blade_shape = ab.add_shape_mesh(
            blade,
            mesh=newton.Mesh(vertices, indices, compute_inertia=False),
            cfg=shape_cfg,
            label=f"{leg}_blade_mesh",
        )
        for robot_shape in robot_shapes:
            ab.add_shape_collision_filter_pair(blade_shape, robot_shape)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(ab, world_count)
    builder.add_ground_plane(cfg=shape_cfg)
    builder.color()
    model = builder.finalize(device=DEVICE)

    fk_state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, fk_state)
    wp.copy(model.body_q, fk_state.body_q)
    wp.copy(model.body_qd, fk_state.body_qd)
    return model


def run(model, world_count):
    solver = SolverVBD(
        model,
        iterations=ITERATIONS,
        rigid_joint_adaptive_stiffness=True,
        rigid_joint_linear_ke=JOINT_KE,
        rigid_joint_angular_ke=JOINT_KE,
        rigid_articulation_solve=ARTICULATION_SOLVE,
        rigid_articulation_relaxation=ARTICULATION_RELAXATION,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    dt = FRAME_DT / SUBSTEPS

    def frame():
        nonlocal state_0, state_1
        model.collide(state_0, contacts)
        for _ in range(SUBSTEPS):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

    for _ in range(WARMUP_FRAMES):
        frame()
    wp.synchronize_device(DEVICE)

    t0 = time.perf_counter()
    for _ in range(TIMED_FRAMES):
        frame()
    wp.synchronize_device(DEVICE)
    wall = (time.perf_counter() - t0) / TIMED_FRAMES * 1e3

    per_kernel = collections.defaultdict(float)
    with wp.ScopedTimer("k", cuda_filter=wp.TIMING_KERNEL, print=False) as timer:
        for _ in range(TIMED_FRAMES):
            frame()
        wp.synchronize_device(DEVICE)
    for r in timer.timing_results:
        per_kernel[r.name.replace("forward kernel ", "").split("<")[0]] += r.elapsed
    per_kernel = {k: v / TIMED_FRAMES for k, v in per_kernel.items()}

    z = state_0.body_q.numpy()[:, 2]
    diag = {
        "contact_count": int(contacts.rigid_contact_count.numpy()[0]),
        "body_z_min": float(np.min(z)),
        "body_z_max": float(np.max(z)),
        "finite": bool(np.all(np.isfinite(state_0.joint_q.numpy()))),
    }
    return {
        "wall_ms_per_frame": wall,
        "gpu_ms_per_frame": sum(per_kernel.values()),
        "kernels": per_kernel,
        "diag": diag,
    }


def main():
    wp.init()
    wp.set_device(DEVICE)
    out = {}
    for world_count in WORLD_COUNTS:
        model = build_model(world_count)
        r = run(model, world_count)
        out[str(world_count)] = r
        print(
            f"worlds={world_count:5d} wall {r['wall_ms_per_frame']:9.3f}  gpu {r['gpu_ms_per_frame']:9.3f}  {r['diag']}",
            flush=True,
        )
        top = sorted(r["kernels"].items(), key=lambda kv: -kv[1])[:6]
        for k, v in top:
            print(f"          {k:58s} {v:9.3f}", flush=True)
        del model

    with open(OUT_JSON, "w") as f:
        json.dump({"tag": TAG, "runs": out}, f, indent=2)


if __name__ == "__main__":
    main()
