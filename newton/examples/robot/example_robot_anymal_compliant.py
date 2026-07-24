# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Anymal Compliant
#
# Spawns the ANYmal-D-Compliant robot from the external compliance repo,
# whose USD authors an n-segment serial spring chain in each shank, and
# steps it under gravity using SolverVBD on the rigid path. The shank
# springs are modeled as position-targeted revolute joints; the leg motors
# are position-targeted at a standing home pose.
#
# Command:
#   python -m newton.examples robot_anymal_compliant \
#       --usd-path /path/to/anymal_d_compliant_svg_xcel_short_exported_n10.usd
#
###########################################################################

import warp as wp

import newton
import newton.examples
from newton import JointTargetMode, JointType
from newton.solvers import SolverVBD

MOTOR_TARGET_KE = 85.0
MOTOR_TARGET_KD = 0.6 * MOTOR_TARGET_KE  # hack to make robot not collapse for solver tuning

SHANK_TARGET_KE = 6000.0  # xcel
SHANK_TARGET_KD = 0.0

CONTACT_KE = 8.0e4
CONTACT_KD = 2.8e4
CONTACT_MU = 0.75

FIXED_PENALTY_STIFFNESS = 1.0e6

SUBSTEPS = 128
ITERATIONS = 30
ARTICULATION_SOLVE = "local"

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
SPRING_SUBSTR = "_shank_spring_"


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.device = wp.get_device()

        articulation_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        articulation_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.0,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        articulation_builder.default_shape_cfg.mu = CONTACT_MU

        articulation_builder.add_usd(
            args.usd_path,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )

        articulation_builder.joint_q[:3] = [0.0, 0.0, BASE_SPAWN_Z]
        if len(articulation_builder.joint_q) > 6:
            articulation_builder.joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]

        n_motor = 0
        n_spring = 0
        for joint_idx in range(articulation_builder.joint_count):
            if articulation_builder.joint_type[joint_idx] != int(JointType.REVOLUTE):
                continue
            short = articulation_builder.joint_label[joint_idx].rsplit("/", 1)[-1]
            q_start = articulation_builder.joint_q_start[joint_idx]
            qd_start = articulation_builder.joint_qd_start[joint_idx]

            if SPRING_SUBSTR in short:
                articulation_builder.joint_target_ke[qd_start] = SHANK_TARGET_KE
                articulation_builder.joint_target_kd[qd_start] = SHANK_TARGET_KD
                articulation_builder.joint_target_mode[qd_start] = int(JointTargetMode.POSITION)
                articulation_builder.joint_target_q[q_start] = 0.0
                n_spring += 1
            elif short.endswith(MOTOR_SUFFIXES):
                home = INITIAL_JOINT_Q.get(short, 0.0)
                articulation_builder.joint_q[q_start] = home
                articulation_builder.joint_target_ke[qd_start] = MOTOR_TARGET_KE
                articulation_builder.joint_target_kd[qd_start] = MOTOR_TARGET_KD
                articulation_builder.joint_target_mode[qd_start] = int(JointTargetMode.POSITION)
                articulation_builder.joint_target_q[q_start] = home
                n_motor += 1

        if not args.quiet:
            print(
                f"[anymal_compliant] joints: {articulation_builder.joint_count} total, "
                f"{n_motor} motor, {n_spring} shank spring"
            )

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_world(articulation_builder)
        builder.add_ground_plane()
        for shape_idx in range(builder.shape_count):
            builder.shape_material_ke[shape_idx] = CONTACT_KE
            builder.shape_material_kd[shape_idx] = CONTACT_KD
            builder.shape_material_mu[shape_idx] = CONTACT_MU
        builder.color()

        self.model = builder.finalize()

        fk_state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, fk_state)
        wp.copy(self.model.body_q, fk_state.body_q)
        wp.copy(self.model.body_qd, fk_state.body_qd)

        self.solver = SolverVBD(
            self.model,
            iterations=ITERATIONS,
            rigid_articulation_solve=ARTICULATION_SOLVE,
            rigid_joint_linear_ke=FIXED_PENALTY_STIFFNESS,
            rigid_joint_angular_ke=FIXED_PENALTY_STIFFNESS,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.contacts = self.model.contacts()
        self.viewer.set_model(self.model)

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
            help="Path to anymal_d_compliant_svg_*_exported_n10.usd from the compliance repo.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
