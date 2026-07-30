# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Anymal Compliant Elastic
#
# Spawns the ANYmal-D-Compliant robot whose USD has the compliant shank blade
# removed (anymal_d_compliant_no_shanks.usd; each leg ends at its *_SHANK
# knee-output link), and rebuilds each leaf-spring blade as a reduced elastic
# body from the blade SVG centerline via ModalGeneratorCurvedBeam. Each blade is
# fixed to its *_SHANK link at the root, and its swept surface mesh both renders
# the deformation and provides the ground contact geometry (filtered against the
# robot's own shapes, which the blade overlaps near the root), so no separate
# foot link is needed. The compliant flex is modelled by floating-frame modal
# coordinates instead of the n-segment serial spring chain. Steps under SolverVBD.
#
# Command:
#   python -m newton.examples robot_anymal_compliant_elastic \
#       --usd-path /path/to/anymal_d_compliant_no_shanks.usd \
#       --centerline /path/to/xcel_exported_centerline.npy
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import JointTargetMode, JointType
from newton.solvers import SolverVBD

MOTOR_TARGET_KE = 85.0
MOTOR_TARGET_KD = 0.6
JOINT_ARMATURE = 0.1

INITIAL_JOINT_Q = {
    "LF_HAA": 0.0,
    "LH_HAA": 0.0,
    "RF_HAA": 0.0,
    "RH_HAA": 0.0,
    "LF_HFE": 0.4,
    "RF_HFE": 0.4,
    "LH_HFE": -0.4,
    "RH_HFE": -0.4,
    "LF_KFE": -0.8,
    "RF_KFE": -0.8,
    "LH_KFE": 0.8,
    "RH_KFE": 0.8,
}

BASE_SPAWN_Z = 1.0
MOTOR_SUFFIXES = ("_HAA", "_HFE", "_KFE")
LEGS = (("LF", True), ("RF", False), ("LH", True), ("RH", False))

BLADE_WIDTH = 0.06
BLADE_DEPTH = 0.01
BLADE_DENSITY = 1600.0
BLADE_YOUNG_MODULUS = 6e10
BLADE_DAMPING_RATIO = 0.01
BLADE_MODE_COUNT = 4

ROOT_OFFSET_X = -0.03
ROOT_OFFSET_Y = 0.05

CONTACT_KE = 8.0e4
CONTACT_KD = 0.35
CONTACT_MU = 0.75

FIXED_PENALTY_STIFFNESS = 1.0e6

SUBSTEPS = 64
ITERATIONS = 128


def _shank_local_nodes(centerline, is_left):
    sign = -1.0 if is_left else 1.0
    root_y = ROOT_OFFSET_Y if is_left else -ROOT_OFFSET_Y
    nodes = np.zeros((centerline.shape[0], 3), dtype=np.float64)
    nodes[:, 0] = ROOT_OFFSET_X
    nodes[:, 1] = root_y + sign * centerline[:, 0]
    nodes[:, 2] = centerline[:, 1]
    return nodes


def _setup_articulation(usd_path):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        armature=JOINT_ARMATURE,
        limit_ke=1.0e3,
        limit_kd=1.0e1,
    )
    builder.default_shape_cfg.mu = CONTACT_MU
    builder.add_usd(
        usd_path,
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )

    builder.joint_q[:3] = [0.0, 0.0, BASE_SPAWN_Z]
    if len(builder.joint_q) > 6:
        builder.joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]

    for joint_idx in range(builder.joint_count):
        if builder.joint_type[joint_idx] != int(JointType.REVOLUTE):
            continue
        short = builder.joint_label[joint_idx].rsplit("/", 1)[-1]
        if not short.endswith(MOTOR_SUFFIXES):
            continue
        q_start = builder.joint_q_start[joint_idx]
        qd_start = builder.joint_qd_start[joint_idx]
        home = INITIAL_JOINT_Q.get(short, 0.0)
        builder.joint_q[q_start] = home
        builder.joint_target_ke[qd_start] = MOTOR_TARGET_KE
        builder.joint_target_kd[qd_start] = MOTOR_TARGET_KD
        builder.joint_target_mode[qd_start] = int(JointTargetMode.POSITION)
        builder.joint_target_pos[qd_start] = home

    return builder


def _shank_world_poses(usd_path):
    probe = _setup_articulation(usd_path)
    model = probe.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    body_q = state.body_q.numpy()
    poses = {}
    for i, label in enumerate(model.body_label):
        short = label.rsplit("/", 1)[-1]
        for leg, _ in LEGS:
            if short == f"{leg}_SHANK":
                poses[leg] = wp.transform(wp.vec3(*body_q[i, :3]), wp.quat(*body_q[i, 3:]))
    return poses


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.device = wp.get_device()

        centerline = np.load(args.centerline)
        if centerline.ndim != 2 or centerline.shape[1] != 2:
            raise ValueError(f"centerline must have shape [points, 2], got {centerline.shape}")

        shank_pose = _shank_world_poses(args.usd_path)

        articulation_builder = _setup_articulation(args.usd_path)
        shank_index = {}
        for body_idx, label in enumerate(articulation_builder.body_label):
            short = label.rsplit("/", 1)[-1]
            for leg, _ in LEGS:
                if short == f"{leg}_SHANK":
                    shank_index[leg] = body_idx

        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.ke = CONTACT_KE
        shape_cfg.kd = CONTACT_KD
        shape_cfg.mu = CONTACT_MU

        robot_shapes = list(range(articulation_builder.shape_count))

        self.blade_bodies = []
        first_frequencies = None
        for leg, is_left in LEGS:
            nodes = _shank_local_nodes(centerline, is_left)
            generator = newton.ModalGeneratorCurvedBeam(
                centerline=nodes,
                width=BLADE_WIDTH,
                depth=BLADE_DEPTH,
                mode_count=BLADE_MODE_COUNT,
                density=BLADE_DENSITY,
                young_modulus=BLADE_YOUNG_MODULUS,
                damping_ratio=BLADE_DAMPING_RATIO,
                label=f"{leg}_blade_basis",
            )
            basis = generator.build()
            if first_frequencies is None:
                first_frequencies = generator.frequencies

            blade = articulation_builder.add_body_elastic(
                xform=shank_pose[leg],
                mass=generator.total_mass,
                com=wp.vec3(*generator.center_of_mass),
                inertia=wp.mat33(*generator.inertia.flatten()),
                modal_basis=basis,
                lock_inertia=True,
                label=f"{leg}_blade",
            )
            self.blade_bodies.append(blade)

            root_local = wp.transform(wp.vec3(*generator.root_local), wp.quat_identity())
            articulation_builder.add_joint_fixed(
                parent=shank_index[leg],
                child=blade,
                parent_xform=root_local,
                child_xform=root_local,
                label=f"{leg}_shank_to_blade",
            )

            vertices, indices = generator.surface_mesh()
            blade_shape = articulation_builder.add_shape_mesh(
                blade,
                mesh=newton.Mesh(vertices, indices, compute_inertia=False),
                cfg=shape_cfg,
                label=f"{leg}_blade_mesh",
            )
            for robot_shape in robot_shapes:
                articulation_builder.add_shape_collision_filter_pair(blade_shape, robot_shape)

        if not args.quiet:
            print(
                f"[anymal_compliant_elastic] blades: {len(self.blade_bodies)}, "
                f"modes/blade: {BLADE_MODE_COUNT}, blade frequencies [Hz]: "
                f"{np.round(first_frequencies, 2).tolist()}"
            )

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_world(articulation_builder)
        builder.add_ground_plane(cfg=shape_cfg)
        builder.color()

        self.model = builder.finalize()

        fk_state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, fk_state)
        wp.copy(self.model.body_q, fk_state.body_q)
        wp.copy(self.model.body_qd, fk_state.body_qd)

        self.solver = SolverVBD(
            self.model,
            iterations=ITERATIONS,
            rigid_joint_adaptive_stiffness=False,
            rigid_joint_linear_ke=FIXED_PENALTY_STIFFNESS,
            rigid_joint_angular_ke=FIXED_PENALTY_STIFFNESS,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.contacts = self.model.contacts()
        self.viewer.set_model(self.model)
        self.viewer.show_elastic_samples = False

    def simulate(self):
        self.model.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        joint_q = self.state_0.joint_q.numpy()
        if not np.all(np.isfinite(joint_q)):
            raise AssertionError("joint_q contains non-finite values")
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > -0.01,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--usd-path",
            type=str,
            required=True,
            help="Path to anymal_d_compliant_no_shanks.usd from the compliance repo.",
        )
        parser.add_argument(
            "--centerline",
            type=str,
            required=True,
            help="Path to the blade SVG centerline .npy (points, 2) from the compliance repo.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
