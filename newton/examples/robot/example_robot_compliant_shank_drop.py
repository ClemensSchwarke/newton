# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Compliant Shank Drop
#
# Contact counterpart of robot_compliant_shank_step. The same two ANYmal-D
# compliant-shank models -- the PRDM segmented revolute spring chain and the
# reduced elastic blade built from the same centerline -- each hang from a
# payload carriage that is constrained to a vertical prismatic joint, and are
# dropped onto the ground under gravity. The step example loads the blade
# through a fixed joint at the tip; this one loads it through a contact, which
# is what the locomotion policy actually experiences.
#
# The constants below are the values the compliance repo trains with on the
# Newton VBD backend (branch `newton`):
#
#   sim.dt = 0.005, num_substeps = 2               -> FPS 200, SUBSTEPS 2
#   VBDSolverCfg(iterations=10,                    -> ITERATIONS
#       rigid_articulation_solve="block_sparse_joints",
#       rigid_joint_linear_ke=1e6, rigid_joint_angular_ke=1e6,
#       rigid_body_contact_buffer_size=256)
#   NewtonShapeCfg(margin=0.001, ke=8e5, kd=1e5, mu=0.8)
#   elastic_env_cfg: default_shape_cfg.kd = 0, blade kd = 2 * 0.35
#
# Contact kd is the mean of the two shapes, so the ground carries kd = 0 and
# each leg carries twice the coefficient its training preset resolves to: the
# PRDM chain reproduces the rigid-foot kd = 1e5 that the compliant task keeps,
# and the blade reproduces the kd = 0.35 that elastic_shank.py derives from
# kd <= min_i c_i / psi_i^2 so a contact cannot overdamp a retained mode.
#
# That asymmetry is the largest single difference between the two robots. At
# the corrected baseline (ARTICULATION_RELAXATION = 0.8, BLADE_YOUNG_MODULUS =
# 9e10, see robot_compliant_shank_step) and 32 substeps, giving both legs
# kd = 0.35 squashes them 12.6 and 14.8 mm and returns the payload to 41.6 and
# 45.6 mm of its 50 mm release height. Giving both the inherited kd = 1e5
# instead converges for neither: the chain returns to 12.7 mm and the blade
# stops compressing at all.
#
# Squash and return height are measured on the payload body, not the blade tip,
# which keeps vibrating in flight.
#
# Command: python -m newton.examples robot_compliant_shank_drop
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import Axis, JointTargetMode
from newton.examples.robot.example_robot_compliant_shank_step import (
    BLADE_DAMPING_RATIO,
    BLADE_DENSITY,
    BLADE_DEPTH,
    BLADE_MODE_COUNT,
    BLADE_WIDTH,
    BLADE_YOUNG_MODULUS,
    N_SEGMENTS,
    SHANK_ARMATURE,
    SHANK_TARGET_KD,
    SHANK_TARGET_KE,
    load_centerline,
    prdm_nodes,
    resolve_centerline,
    segment_quat,
    visual_shape_config,
)
from newton.solvers import SolverVBD

PAYLOAD_MASS = 12.5
DROP_HEIGHT = 0.05
LEG_SPACING = 1.0
DURATION = 0.7

CONTACT_KE = 8.0e5
CONTACT_MU = 0.8
CONTACT_MARGIN = 0.001
GROUND_KD = 0.0
BLADE_CONTACT_KD = 0.35
PRDM_CONTACT_KD = 1.0e5

FPS = 200
SUBSTEPS = 2
ITERATIONS = 10
JOINT_PENALTY_KE = 1.0e6
ARTICULATION_SOLVE = "block_sparse_joints"
ARTICULATION_RELAXATION = 0.65
BODY_CONTACT_BUFFER = 256
RIGID_CONTACT_MAX = 8192

PLOT_PATH = "compliant_shank_drop.png"


def contact_shape_config(kd: float) -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.ke = CONTACT_KE
    cfg.kd = kd
    cfg.mu = CONTACT_MU
    cfg.margin = CONTACT_MARGIN
    cfg.gap = 0.0
    return cfg


def box_corners(length: float) -> np.ndarray:
    return np.array(
        [
            [sx * 0.5 * BLADE_WIDTH, sy * 0.5 * BLADE_DEPTH, sz * 0.5 * length - 0.5 * length]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )


def prdm_lowest_z(nodes: np.ndarray) -> float:
    lowest = 0.0
    for i in range(1, nodes.shape[0]):
        segment = nodes[i] - nodes[i - 1]
        length = float(np.linalg.norm(segment))
        frame = wp.transform(wp.vec3(*nodes[i - 1]), segment_quat(segment))
        for corner in box_corners(length):
            world = wp.transform_point(frame, wp.vec3(*corner))
            lowest = min(lowest, float(world[2]))
    return lowest


class Example:
    def __init__(self, viewer, args):
        self.fps = FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer

        self.centerline_path = resolve_centerline(getattr(args, "centerline", None))
        centerline = load_centerline(self.centerline_path)
        chain_nodes = prdm_nodes(centerline, N_SEGMENTS)

        builder = newton.ModelBuilder(up_axis=Axis.Z)
        builder.num_rigid_contacts_per_world = RIGID_CONTACT_MAX
        self.visual_cfg = visual_shape_config()
        self.ground_shape = builder.add_ground_plane(cfg=contact_shape_config(GROUND_KD))

        self.generator = newton.ModalGeneratorCurvedBeam(
            centerline=centerline,
            width=BLADE_WIDTH,
            depth=BLADE_DEPTH,
            plane="yz",
            mode_count=BLADE_MODE_COUNT,
            density=BLADE_DENSITY,
            young_modulus=BLADE_YOUNG_MODULUS,
            damping_ratio=BLADE_DAMPING_RATIO,
            label="blade_basis",
        )
        basis = self.generator.build()
        self.frequencies = np.asarray(self.generator.frequencies, dtype=np.float64)
        self.mode_stiffness = np.asarray(basis.mode_stiffness, dtype=np.float64)

        elastic_bottom = float(np.min(np.asarray(basis.sample_points, dtype=np.float64)[:, 2]))
        chain_bottom = prdm_lowest_z(chain_nodes)

        self.prdm_carriage, self.prdm_bodies = self._build_prdm_leg(
            builder, chain_nodes, -0.5 * LEG_SPACING, -chain_bottom + DROP_HEIGHT
        )
        self.elastic_carriage, self.elastic_body = self._build_elastic_leg(
            builder, basis, 0.5 * LEG_SPACING, -elastic_bottom + DROP_HEIGHT
        )

        builder.color()
        self.model = builder.finalize()
        self.model.rigid_contact_max = RIGID_CONTACT_MAX

        fk_state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, fk_state)
        wp.copy(self.model.body_q, fk_state.body_q)
        wp.copy(self.model.body_qd, fk_state.body_qd)

        self.solver = SolverVBD(
            self.model,
            iterations=ITERATIONS,
            rigid_joint_linear_ke=JOINT_PENALTY_KE,
            rigid_joint_angular_ke=JOINT_PENALTY_KE,
            rigid_articulation_solve=ARTICULATION_SOLVE,
            rigid_articulation_relaxation=ARTICULATION_RELAXATION,
            rigid_body_contact_buffer_size=BODY_CONTACT_BUFFER,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        joint_q_start = self.model.joint_q_start.numpy()
        self.spring_q_starts = np.array([int(joint_q_start[j]) for j in self.prdm_spring_joints], dtype=np.int64)
        elastic_index = int(self.model.body_elastic_index.numpy()[self.elastic_body])
        owner = int(self.model.elastic_joint.numpy()[elastic_index])
        self.modal_q_start = int(joint_q_start[owner]) + 7

        self.ik_q = wp.zeros(self.model.joint_coord_count, dtype=float)
        self.ik_qd = wp.zeros(self.model.joint_dof_count, dtype=float)
        self.rest_spring_q = self._spring_angles()

        self.rest_carriage_z = self.state_0.body_q.numpy()[[self.prdm_carriage, self.elastic_carriage], 2].copy()
        self.trace = {
            "t": [],
            "prdm_z": [],
            "elastic_z": [],
            "prdm_vz": [],
            "elastic_vz": [],
            "prdm_energy": [],
            "elastic_energy": [],
        }

        self.viewer.set_model(self.model)
        self.viewer.show_elastic_strain = True
        self.viewer.set_camera(pos=wp.vec3(2.6, 0.0, 0.55), pitch=-6.0, yaw=180.0)

    def _build_prdm_leg(self, builder, nodes, x, carriage_z):
        contact_cfg = contact_shape_config(2.0 * PRDM_CONTACT_KD)
        carriage_pose = wp.transform(wp.vec3(x, 0.0, carriage_z), wp.quat_identity())
        carriage = builder.add_link(
            xform=carriage_pose,
            mass=PAYLOAD_MASS,
            inertia=wp.mat33(0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05),
            label="prdm_payload",
        )
        builder.add_shape_box(carriage, hx=0.05, hy=0.05, hz=0.05, cfg=self.visual_cfg, label="prdm_payload_box")
        slider = builder.add_joint_prismatic(
            parent=-1,
            child=carriage,
            parent_xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
            axis=Axis.Z,
            label="prdm_slider",
        )
        builder.joint_q[builder.joint_q_start[slider]] = carriage_z

        joints = [slider]
        self.prdm_spring_joints = []
        bodies = []
        shapes = []
        parent = carriage
        prev_world = carriage_pose
        for i in range(1, nodes.shape[0]):
            segment = nodes[i] - nodes[i - 1]
            length = float(np.linalg.norm(segment))
            world = wp.transform_multiply(carriage_pose, wp.transform(wp.vec3(*nodes[i - 1]), segment_quat(segment)))

            mass = BLADE_DENSITY * BLADE_WIDTH * BLADE_DEPTH * length
            ixx = mass / 12.0 * (BLADE_DEPTH**2 + length**2)
            iyy = mass / 12.0 * (BLADE_WIDTH**2 + length**2)
            izz = mass / 12.0 * (BLADE_WIDTH**2 + BLADE_DEPTH**2)
            body = builder.add_link(
                xform=world,
                com=wp.vec3(0.0, 0.0, -0.5 * length),
                mass=mass,
                inertia=wp.mat33(ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz),
                label=f"prdm_seg_{i}",
            )
            cfg = self.visual_cfg if i == 1 else contact_cfg
            shape = builder.add_shape_box(
                body,
                xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * length), wp.quat_identity()),
                hx=0.5 * BLADE_WIDTH,
                hy=0.5 * BLADE_DEPTH,
                hz=0.5 * length,
                cfg=cfg,
                label=f"prdm_seg_{i}_box",
            )
            for other in shapes:
                builder.add_shape_collision_filter_pair(shape, other)
            shapes.append(shape)

            joint_parent_xform = wp.transform_multiply(wp.transform_inverse(prev_world), world)
            joint = builder.add_joint_revolute(
                parent=parent,
                child=body,
                axis=(1.0, 0.0, 0.0),
                parent_xform=joint_parent_xform,
                child_xform=wp.transform_identity(),
                target_pos=0.0,
                target_ke=SHANK_TARGET_KE,
                target_kd=SHANK_TARGET_KD,
                armature=SHANK_ARMATURE,
                actuator_mode=JointTargetMode.POSITION,
                label=f"prdm_spring_{i}",
            )
            joints.append(joint)
            self.prdm_spring_joints.append(joint)
            bodies.append(body)
            parent = body
            prev_world = world

        builder.add_articulation(joints, label="prdm_leg")
        return carriage, bodies

    def _build_elastic_leg(self, builder, basis, x, carriage_z):
        contact_cfg = contact_shape_config(2.0 * BLADE_CONTACT_KD)
        carriage_pose = wp.transform(wp.vec3(x, 0.0, carriage_z), wp.quat_identity())
        carriage = builder.add_link(
            xform=carriage_pose,
            mass=PAYLOAD_MASS,
            inertia=wp.mat33(0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05),
            label="elastic_payload",
        )
        builder.add_shape_box(carriage, hx=0.05, hy=0.05, hz=0.05, cfg=self.visual_cfg, label="elastic_payload_box")
        slider = builder.add_joint_prismatic(
            parent=-1,
            child=carriage,
            parent_xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
            axis=Axis.Z,
            label="elastic_slider",
        )
        builder.joint_q[builder.joint_q_start[slider]] = carriage_z

        blade = builder.add_link_elastic(
            xform=carriage_pose,
            mass=self.generator.total_mass,
            com=wp.vec3(*self.generator.center_of_mass),
            inertia=wp.mat33(*self.generator.inertia.flatten()),
            modal_basis=basis,
            lock_inertia=True,
            label="elastic_blade",
        )
        joints = [slider, builder.body_elastic_joint[blade]]

        vertices, indices = self.generator.surface_mesh()
        builder.add_shape_mesh(
            blade,
            mesh=newton.Mesh(vertices, indices, compute_inertia=False),
            cfg=contact_cfg,
            label="elastic_blade_mesh",
        )

        root_local = wp.transform(wp.vec3(*self.generator.root_local), wp.quat_identity())
        joints.append(
            builder.add_joint_fixed(
                parent=carriage,
                child=blade,
                parent_xform=root_local,
                child_xform=root_local,
                label="elastic_clamp",
            )
        )
        builder.add_articulation(joints, label="elastic_leg", allow_closed_loops=True)
        return carriage, blade

    def _spring_angles(self):
        newton.eval_ik(self.model, self.state_0, self.ik_q, self.ik_qd)
        return self.ik_q.numpy()[self.spring_q_starts].copy()

    def _sample(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        spring_q = self._spring_angles() - self.rest_spring_q
        modal_q = self.state_0.joint_q.numpy()[self.modal_q_start : self.modal_q_start + BLADE_MODE_COUNT]
        self.trace["t"].append(self.sim_time)
        self.trace["prdm_z"].append(float(body_q[self.prdm_carriage, 2]))
        self.trace["elastic_z"].append(float(body_q[self.elastic_carriage, 2]))
        self.trace["prdm_vz"].append(float(body_qd[self.prdm_carriage, 2]))
        self.trace["elastic_vz"].append(float(body_qd[self.elastic_carriage, 2]))
        self.trace["prdm_energy"].append(0.5 * float(SHANK_TARGET_KE * np.sum(spring_q**2)))
        self.trace["elastic_energy"].append(0.5 * float(np.sum(self.mode_stiffness * modal_q**2)))

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt
            self._sample()

    def step(self):
        self.simulate()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def payload_force(self, key: str) -> np.ndarray:
        vz = np.asarray(self.trace[f"{key}_vz"])
        return PAYLOAD_MASS * (np.gradient(vz, self.sim_dt) + 9.81)

    def summary(self):
        rows = []
        for index, (name, key) in enumerate((("PRDM", "prdm"), ("elastic", "elastic"))):
            z = np.asarray(self.trace[f"{key}_z"])
            energy = np.asarray(self.trace[f"{key}_energy"])
            force = self.payload_force(key)
            bottom = int(np.argmin(z))
            peak = float(np.max(z[bottom:])) if bottom < z.size - 1 else float(z[bottom])
            rebound_height = DROP_HEIGHT - (self.rest_carriage_z[index] - peak)
            rows.append(
                {
                    "name": name,
                    "compression": float(self.rest_carriage_z[index] - DROP_HEIGHT - z[bottom]),
                    "lowest_z": float(z[bottom]),
                    "rebound_height": float(rebound_height),
                    "peak_energy": float(np.max(energy)),
                    "final_energy": float(energy[-1]),
                    "peak_force": float(np.max(force)),
                    "final_z": float(z[-1]),
                }
            )
        return rows

    def print_summary(self):
        print(f"[compliant_shank_drop] centerline: {self.centerline_path}")
        print(f"[compliant_shank_drop] blade modes [Hz]: {np.round(self.frequencies, 2).tolist()}")
        print(
            f"[compliant_shank_drop] payload {PAYLOAD_MASS} kg, drop {DROP_HEIGHT} m, "
            f"dt={self.sim_dt * 1e3:.2f} ms, iterations={ITERATIONS}"
        )
        print(
            f"{'model':>10} {'contact kd':>11} {'min z [m]':>10} {'squash [m]':>11} "
            f"{'return [m]':>12} {'peak E [J]':>11} {'peak F [N]':>11} {'final z [m]':>12}"
        )
        for index, row in enumerate(self.summary()):
            kd = PRDM_CONTACT_KD if index == 0 else BLADE_CONTACT_KD
            print(
                f"{row['name']:>10} {kd:>11.4g} {row['lowest_z']:>10.4f} {row['compression']:>11.4f} "
                f"{row['rebound_height']:>12.4f} {row['peak_energy']:>11.3f} {row['peak_force']:>11.1f} "
                f"{row['final_z']:>12.4f}"
            )

    def plot(self, path: str):
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        t = np.asarray(self.trace["t"])
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8.0, 9.0))
        for index, (name, key) in enumerate((("PRDM chain", "prdm"), ("reduced elastic", "elastic"))):
            drop = np.asarray(self.trace[f"{key}_z"]) - self.rest_carriage_z[index]
            axes[0].plot(t, drop, label=name)
            axes[1].plot(t, self.trace[f"{key}_energy"], label=name)
            axes[2].plot(t, self.payload_force(key), label=name)
        axes[0].axhline(-DROP_HEIGHT, color="k", ls="--", lw=1.0, label="blade touches ground")
        axes[2].axhline(PAYLOAD_MASS * 9.81, color="k", ls="--", lw=1.0, label="payload weight")
        axes[0].set_ylabel("payload drop [m]")
        axes[1].set_ylabel("energy stored in the leg [J]")
        axes[2].set_ylabel("force on the payload [N]")
        axes[2].set_xlabel("time [s]")
        for axis in axes:
            axis.grid(alpha=0.3)
            axis.legend(fontsize=8)
        fig.suptitle(f"Compliant shank drop: {PAYLOAD_MASS:.1f} kg payload from {DROP_HEIGHT * 100:.0f} cm")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        print(f"[compliant_shank_drop] wrote {path}")

    def test_final(self):
        if not np.all(np.isfinite(self.state_0.joint_q.numpy())):
            raise AssertionError("joint_q contains non-finite values")
        for row in self.summary():
            if row["lowest_z"] <= 0.0:
                raise AssertionError(f"{row['name']} payload fell through the ground: {row['lowest_z']}")
            if row["peak_energy"] <= 0.0:
                raise AssertionError(f"{row['name']} leg stored no energy on impact")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--centerline",
            type=str,
            default=None,
            help="Blade centerline .npy of shape [points, 2]. Defaults to the traced ANYmal-D "
            "xcel-short blade from the compliance repo (override its location with COMPLIANCE_ROOT).",
        )
        return parser


def test(device=None):
    import newton.viewer  # noqa: PLC0415

    parser = Example.create_parser()
    args = parser.parse_args([])
    args.test = True
    frames = int(DURATION * FPS)
    with wp.ScopedDevice(device):
        example = Example(newton.viewer.ViewerNull(num_frames=frames), args)
        for _ in range(frames):
            example.step()
        example.test_final()
    return example


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
    example.print_summary()
    example.plot(PLOT_PATH)
