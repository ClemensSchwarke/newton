# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Compliant Shank Step
#
# Side-by-side comparison of the two ANYmal-D compliant-shank models under an
# identical tip load. One leaf-spring blade is the PRDM segmented revolute
# spring chain that the training USD authors, the other is a reduced elastic
# body built from the same centerline by ModalGeneratorCurvedBeam. Both are
# clamped to the world at the root, gravity is off, and a vertical (+Z) force
# at each tip is zero until STEP_TIME and then jumps to TIP_FORCE_Z, so the
# transient and steady tip deflection of the two models can be compared
# directly.
#
# Every solver, material and discretization constant below is the value the
# compliance repo trains with on the Newton VBD backend (branch `newton`):
#
#   sim.dt = 0.005, num_substeps = 2               -> FPS 200, SUBSTEPS 2
#   VBDSolverCfg(iterations=10,                    -> ITERATIONS
#       rigid_articulation_solve="block_sparse_joints",
#       rigid_joint_linear_ke=1e6, rigid_joint_angular_ke=1e6)
#   compliant_shanks ImplicitActuatorCfg(stiffness=6551, damping=2)
#   elastic_shank.py: E=6e10, damping_ratio=0.01, 4 modes, 1600 kg/m^3,
#       width 0.06 m, depth 0.01 m, xcel-short centerline
#
# The chain and the blade each live in one articulation so the block-sparse
# articulation solve sees every joint: joints left outside an articulation are
# silently dropped by that solve mode, which is why the segments are built with
# add_link/add_articulation rather than add_body.
#
# The summary prints each model's settled tip deflection against two references.
# The first is the static solution of that same model -- the nonlinear equilibrium
# of the rigid-link chain, and the retained modes' response to the same tip force.
# Both are solved here in closed form, so that percentage is solver error with
# modelling error factored out, and at the training preset neither model reaches
# its own answer: ARTICULATION_RELAXATION = 0.65 is a pathological point at 10
# iterations and leaves the chain at ~35%, while the blade needs a far smaller
# step than 2.5 ms.
#
# The second reference is measured rather than modelled. ABAQUS_TIP_DEFLECTION is
# the tip displacement of a 3D solid Abaqus model of the same blade -- xcel-short
# at 90 GPa, 0.06 x 0.01 m section, base encastre, ABAQUS_TIP_FORCE_Z at a
# kinematically coupled tip -- and is the closest thing here to what the real
# blade does. It is only valid at that load, so it is reported only when
# TIP_FORCE_Z matches.
#
# Command: python -m newton.examples robot_compliant_shank_step
#
###########################################################################

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import Axis, JointTargetMode
from newton.solvers import SolverVBD

COMPLIANCE_ROOT = Path(os.environ.get("COMPLIANCE_ROOT", Path.home() / "git" / "isaac" / "compliance"))
CENTERLINE_CANDIDATES = (
    COMPLIANCE_ROOT
    / "source"
    / "compliance"
    / "compliance"
    / "assets"
    / "data"
    / "Anymal-D-Compliant"
    / "xcel_short_exported_centerline.npy",
    Path.home()
    / "ETH"
    / "PhD"
    / "Projects"
    / "Compliance"
    / "CAD"
    / "ossur_svgs"
    / "xcel_short_exported_centerline.npy",
)

CLAMP_Z = 0.8
CLAMP_Y = 0.25
STEP_TIME = 0.05
DURATION = 1.05
TIP_FORCE_Z = 1000.0
ABAQUS_TIP_FORCE_Z = 1000.0
ABAQUS_TIP_DEFLECTION = (0.0788, 0.1044)

N_SEGMENTS = 10
BLADE_WIDTH = 0.06
BLADE_DEPTH = 0.01
BLADE_DENSITY = 1600.0

BLADE_YOUNG_MODULUS = 6.0e10
BLADE_DAMPING_RATIO = 0.01
BLADE_MODE_COUNT = 4

SHANK_TARGET_KE = 6551.0
SHANK_TARGET_KD = 2.0
SHANK_ARMATURE = 0.1

FPS = 200
SUBSTEPS = 2
ITERATIONS = 10
JOINT_PENALTY_KE = 1.0e6
ARTICULATION_SOLVE = "block_sparse_joints"
ARTICULATION_RELAXATION = 0.65

PLOT_PATH = "compliant_shank_step.png"


def resolve_centerline(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"centerline not found: {path}")
        return path
    for candidate in CENTERLINE_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "no blade centerline found; pass --centerline or set COMPLIANCE_ROOT. Looked in: "
        + ", ".join(str(c) for c in CENTERLINE_CANDIDATES)
    )


def load_centerline(path: Path) -> np.ndarray:
    centerline = np.load(path)
    if centerline.ndim != 2 or centerline.shape[1] != 2:
        raise ValueError(f"centerline must have shape [points, 2], got {centerline.shape}")
    return centerline


def prdm_nodes(centerline: np.ndarray, n_segments: int) -> np.ndarray:
    points = np.asarray(centerline, dtype=np.float64)
    segment_length = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segment_length)])
    targets = np.linspace(0.0, arc[-1], n_segments + 1)
    resampled = np.column_stack([np.interp(targets, arc, points[:, 0]), np.interp(targets, arc, points[:, 1])])
    tangents = np.diff(resampled, axis=0)
    abs_angles = np.arctan2(tangents[:, 1], tangents[:, 0])
    link_length = float(np.mean(np.linalg.norm(tangents, axis=1)))
    nodes = np.zeros((n_segments + 1, 3), dtype=np.float64)
    for i in range(1, n_segments + 1):
        angle = abs_angles[i - 1]
        nodes[i] = nodes[i - 1] + link_length * np.array([0.0, np.cos(angle), np.sin(angle)])
    return nodes


def segment_quat(direction: np.ndarray) -> wp.quat:
    z_axis = -direction / np.linalg.norm(direction)
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    basis = np.column_stack([x_axis, y_axis, z_axis])
    return wp.quat_from_matrix(wp.mat33(*basis.flatten()))


def chain_static_deflection(nodes: np.ndarray, stiffness: float, force: float) -> np.ndarray:
    """Nonlinear static tip deflection [m] of the rigid-link chain under a vertical tip force."""
    planar = nodes[:, 1:]
    segments = np.diff(planar, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    rest = np.arctan2(segments[:, 1], segments[:, 0])

    def tip(theta):
        angles = rest + np.cumsum(theta)
        return np.array([np.sum(lengths * np.cos(angles)), np.sum(lengths * np.sin(angles))])

    theta = np.zeros(lengths.size)
    for _ in range(2000):
        angles = rest + np.cumsum(theta)
        arms = np.cumsum((lengths * np.cos(angles))[::-1])[::-1]
        residual = force * arms / stiffness - theta
        theta = theta + 0.5 * residual
        if np.max(np.abs(residual)) < 1.0e-12:
            break
    return tip(theta) - tip(np.zeros(lengths.size))


def modal_static_deflection(generator, basis, force: float) -> np.ndarray:
    """Static tip deflection [m] the retained modes can represent under a vertical tip force."""
    points = np.asarray(basis.sample_points, dtype=np.float64)
    tip = int(np.argmin(np.linalg.norm(points - np.asarray(generator.tip_local, dtype=np.float64), axis=1)))
    phi = np.asarray(basis.sample_phi, dtype=np.float64)[tip]
    stiffness = np.asarray(basis.mode_stiffness, dtype=np.float64)
    return (phi.T @ (phi[:, 2] * force / stiffness))[1:]


def visual_shape_config() -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.has_shape_collision = False
    cfg.has_particle_collision = False
    cfg.density = 0.0
    return cfg


def ring_metrics(times: np.ndarray, deflection: np.ndarray, step_time: float):
    tt = times[times >= step_time]
    seg = deflection[times >= step_time]
    if seg.size < 8:
        return None
    tail = seg[3 * seg.size // 4 :]
    disp = float(tail.mean())
    settled = bool(float(np.std(tail)) <= 0.02 * max(abs(disp), 1e-9))
    osc = seg - disp
    peaks = [
        k
        for k in range(1, osc.size - 1)
        if osc[k] > osc[k - 1] and osc[k] > osc[k + 1] and abs(osc[k]) > 0.02 * abs(disp)
    ]
    freq = 0.0
    zeta = float("nan")
    if len(peaks) >= 2:
        freq = 1.0 / float(np.mean(np.diff(tt[np.array(peaks)])))
        cycles = min(4, len(peaks) - 1)
        first, last = abs(osc[peaks[0]]), abs(osc[peaks[cycles]])
        if first > 0.0 and last > 0.0:
            delta = np.log(first / last) / cycles
            zeta = delta / np.sqrt(4.0 * np.pi**2 + delta**2)
    return disp, freq, zeta, len(peaks), settled


class Example:
    def __init__(self, viewer, args):
        self.fps = FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        self.centerline_path = resolve_centerline(getattr(args, "centerline", None))
        centerline = load_centerline(self.centerline_path)
        chain_nodes = prdm_nodes(centerline, N_SEGMENTS)

        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=0.0)
        self.visual_cfg = visual_shape_config()

        prdm_root = wp.transform(wp.vec3(0.0, -CLAMP_Y, CLAMP_Z), wp.quat_identity())
        elastic_root = wp.transform(wp.vec3(0.0, CLAMP_Y, CLAMP_Z), wp.quat_identity())

        self.prdm_body, self.prdm_tip_local = self._build_prdm_shank(builder, chain_nodes, prdm_root)
        self.elastic_body = self._build_elastic_shank(builder, centerline, elastic_root)
        self.reference = {
            "PRDM": chain_static_deflection(chain_nodes, SHANK_TARGET_KE, TIP_FORCE_Z),
            "elastic": modal_static_deflection(self.generator, self.basis, TIP_FORCE_Z),
        }
        self.true_reference = np.array(ABAQUS_TIP_DEFLECTION) if TIP_FORCE_Z == ABAQUS_TIP_FORCE_Z else None

        builder.color()
        self.model = builder.finalize()

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
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        elastic_index = int(self.model.body_elastic_index.numpy()[self.elastic_body])
        self.elastic_owner = int(self.model.elastic_joint.numpy()[elastic_index])
        self.elastic_q_start = int(self.model.joint_q_start.numpy()[self.elastic_owner]) + 7
        self.elastic_qd_start = int(self.model.joint_qd_start.numpy()[self.elastic_owner]) + 6
        points = np.asarray(self.basis.sample_points, dtype=np.float64)
        tip_sample = int(np.argmin(np.linalg.norm(points - self.elastic_tip_local, axis=1)))
        self.elastic_tip_phi = np.asarray(self.basis.sample_phi, dtype=np.float64)[tip_sample]
        self.body_com = self.model.body_com.numpy().astype(np.float64)
        self.body_force = np.zeros((self.model.body_count, 6), dtype=np.float32)
        self.joint_force = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        self.rest_tip = self._tip_positions()
        self.trace = {"t": [], "prdm": [], "elastic": []}

        self.viewer.set_model(self.model)
        self.viewer.show_elastic_strain = True
        self._frame_camera(chain_nodes, centerline)

    def _frame_camera(self, chain_nodes, centerline):
        elastic_nodes = np.zeros((centerline.shape[0], 3), dtype=np.float64)
        elastic_nodes[:, 1] = centerline[:, 0]
        elastic_nodes[:, 2] = centerline[:, 1]
        world_nodes = np.vstack(
            [
                chain_nodes + np.array([0.0, -CLAMP_Y, CLAMP_Z]),
                elastic_nodes + np.array([0.0, CLAMP_Y, CLAMP_Z]),
            ]
        )
        lo = world_nodes.min(axis=0) - 0.1
        hi = world_nodes.max(axis=0) + 0.1
        center = 0.5 * (lo + hi)
        extent = float(np.max(hi - lo))
        distance = extent / (2.0 * np.tan(np.radians(22.5))) * 1.05
        self.viewer.set_camera(wp.vec3(center[0] + distance, center[1], center[2]), 0.0, 180.0)

    def _build_prdm_shank(self, builder, nodes, root_xform):
        parent = -1
        joints = []
        prev_body = -1
        prev_world = wp.transform_identity()
        tip_local = wp.vec3(0.0, 0.0, 0.0)
        for i in range(1, nodes.shape[0]):
            segment = nodes[i] - nodes[i - 1]
            length = float(np.linalg.norm(segment))
            world = wp.transform_multiply(root_xform, wp.transform(wp.vec3(*nodes[i - 1]), segment_quat(segment)))

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
            builder.add_shape_box(
                body,
                xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * length), wp.quat_identity()),
                hx=0.5 * BLADE_WIDTH,
                hy=0.5 * BLADE_DEPTH,
                hz=0.5 * length,
                cfg=self.visual_cfg,
                label=f"prdm_seg_{i}_box",
            )

            joint_parent_xform = wp.transform_multiply(wp.transform_inverse(prev_world), world)
            joints.append(
                builder.add_joint_revolute(
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
            )

            parent = body
            prev_body = body
            prev_world = world
            tip_local = wp.vec3(0.0, 0.0, -length)

        builder.add_articulation(joints, label="prdm_shank")
        return prev_body, np.array([tip_local[0], tip_local[1], tip_local[2]], dtype=np.float64)

    def _build_elastic_shank(self, builder, centerline, root_xform):
        generator = newton.ModalGeneratorCurvedBeam(
            centerline=centerline,
            width=BLADE_WIDTH,
            depth=BLADE_DEPTH,
            plane="yz",
            mode_count=BLADE_MODE_COUNT,
            density=BLADE_DENSITY,
            young_modulus=BLADE_YOUNG_MODULUS,
            damping_ratio=BLADE_DAMPING_RATIO,
            label="elastic_shank_basis",
        )
        basis = generator.build()
        self.generator = generator
        self.basis = basis
        self.elastic_tip_local = np.asarray(generator.tip_local, dtype=np.float64)
        self.frequencies = np.asarray(generator.frequencies, dtype=np.float64)
        blade = builder.add_link_elastic(
            xform=root_xform,
            mass=generator.total_mass,
            com=wp.vec3(*generator.center_of_mass),
            inertia=wp.mat33(*generator.inertia.flatten()),
            modal_basis=basis,
            lock_inertia=True,
            label="elastic_shank",
        )
        joints = [builder.body_elastic_joint[blade]]

        vertices, indices = generator.surface_mesh()
        builder.add_shape_mesh(
            blade,
            mesh=newton.Mesh(vertices, indices, compute_inertia=False),
            cfg=self.visual_cfg,
            label="elastic_shank_mesh",
        )

        root_local = wp.transform(wp.vec3(*generator.root_local), wp.quat_identity())
        joints.append(
            builder.add_joint_fixed(
                parent=-1,
                child=blade,
                parent_xform=wp.transform_multiply(root_xform, root_local),
                child_xform=root_local,
                label="elastic_clamp",
            )
        )

        builder.add_articulation(joints, label="elastic_shank", allow_closed_loops=True)
        return blade

    def _tip_frames(self):
        body_q = self.state_0.body_q.numpy().astype(np.float64)
        modal_q = self.state_0.joint_q.numpy()[self.elastic_q_start : self.elastic_q_start + BLADE_MODE_COUNT]
        frames = []
        for body, local in (
            (self.prdm_body, self.prdm_tip_local),
            (self.elastic_body, self.elastic_tip_local + self.elastic_tip_phi.T @ modal_q),
        ):
            transform = wp.transform(*body_q[body])
            rotation = np.array(wp.quat_to_matrix(wp.transform_get_rotation(transform)), dtype=np.float64).reshape(3, 3)
            origin = body_q[body, :3]
            frames.append((rotation, origin + rotation @ local, origin + rotation @ self.body_com[body]))
        return frames

    def _tip_positions(self):
        return np.array([point for _, point, _ in self._tip_frames()])

    def _apply_tip_forces(self):
        force = np.array([0.0, 0.0, TIP_FORCE_Z], dtype=np.float64)
        self.body_force[:] = 0.0
        self.joint_force[:] = 0.0
        for body, (rotation, point, com) in zip(
            (self.prdm_body, self.elastic_body), self._tip_frames(), strict=True
        ):
            self.body_force[body, :3] = force
            self.body_force[body, 3:] = np.cross(point - com, force)
            if body == self.elastic_body:
                self.joint_force[self.elastic_qd_start : self.elastic_qd_start + BLADE_MODE_COUNT] = (
                    self.elastic_tip_phi @ (rotation.T @ force)
                )
        self.state_0.body_f.assign(self.body_force.reshape(-1))
        self.control.joint_f.assign(self.joint_force)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.control.joint_f.zero_()
            if self.sim_time >= STEP_TIME:
                self._apply_tip_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt
            tips = self._tip_positions()
            self.trace["t"].append(self.sim_time)
            self.trace["prdm"].append(tips[0])
            self.trace["elastic"].append(tips[1])

    def step(self):
        self.simulate()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def summary(self):
        times = np.asarray(self.trace["t"])
        rows = []
        for index, name in enumerate(("PRDM", "elastic")):
            key = "prdm" if index == 0 else "elastic"
            deflection = np.asarray(self.trace[key])[:, 2] - self.rest_tip[index, 2]
            metrics = ring_metrics(times, deflection, STEP_TIME)
            if metrics is None:
                continue
            disp, freq, zeta, peaks, settled = metrics
            reference = float(self.reference[name][1])
            rows.append(
                {
                    "name": name,
                    "disp": disp,
                    "reference": reference,
                    "freq": freq,
                    "zeta": zeta,
                    "peaks": peaks,
                    "settled": settled,
                }
            )
        return rows

    def print_summary(self):
        print(f"[compliant_shank_step] centerline: {self.centerline_path}")
        print(f"[compliant_shank_step] blade modes [Hz]: {np.round(self.frequencies, 2).tolist()}")
        print(
            f"[compliant_shank_step] dt={self.sim_dt * 1e3:.2f} ms, iterations={ITERATIONS}, "
            f"articulation_solve={ARTICULATION_SOLVE}, relaxation={ARTICULATION_RELAXATION}"
        )
        if self.true_reference is not None:
            print(f"[compliant_shank_step] Abaqus true static disp_z: {self.true_reference[1]:+.5f} m")
        print(
            f"{'model':>10} {'disp_z [m]':>12} {'model [m]':>11} {'of model':>10} {'of true':>9} {'f [Hz]':>9} {'zeta':>9}"
        )
        for row in self.summary():
            zeta = f"{row['zeta']:.4f}" if np.isfinite(row["zeta"]) else "n/a"
            note = "" if row["settled"] else "  (not settled)"
            fraction = row["disp"] / row["reference"] * 100.0
            true = (
                f"{row['disp'] / float(self.true_reference[1]) * 100.0:>7.1f}%"
                if self.true_reference is not None
                else f"{'n/a':>8}"
            )
            print(
                f"{row['name']:>10} {row['disp']:>12.5f} {row['reference']:>11.5f} {fraction:>9.1f}% "
                f"{true:>9} {row['freq']:>9.2f} {zeta:>9}{note}"
            )

    def plot(self, path: str):
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        times = np.asarray(self.trace["t"])
        prdm = np.asarray(self.trace["prdm"]) - self.rest_tip[0]
        elastic = np.asarray(self.trace["elastic"]) - self.rest_tip[1]
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8.0, 6.0))
        for axis, index, name in zip(axes, (1, 2), ("y", "z"), strict=True):
            axis.plot(times, prdm[:, index], color="tab:blue", label=f"PRDM chain ({N_SEGMENTS} segments)")
            axis.plot(times, elastic[:, index], color="tab:orange", label=f"reduced elastic ({BLADE_MODE_COUNT} modes)")
            axis.axhline(
                self.reference["PRDM"][index - 1], color="tab:blue", ls=":", lw=1.4, label="PRDM model equilibrium"
            )
            axis.axhline(
                self.reference["elastic"][index - 1],
                color="tab:orange",
                ls="--",
                lw=1.4,
                label="elastic model equilibrium",
            )
            if self.true_reference is not None:
                axis.axhline(self.true_reference[index - 1], color="black", ls="-.", lw=1.6, label="Abaqus true static")
            axis.set_ylabel(f"tip {name} deflection [m]")
            axis.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel("time [s]")
        fig.suptitle(f"Compliant shank tip response to a {TIP_FORCE_Z:.0f} N step at the tip")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        print(f"[compliant_shank_step] wrote {path}")

    def test_final(self):
        if not np.all(np.isfinite(self.state_0.joint_q.numpy())):
            raise AssertionError("joint_q contains non-finite values")
        rows = self.summary()
        if len(rows) != 2:
            raise AssertionError("expected ring-down metrics for both models")
        for row in rows:
            if not 0.0 < row["disp"] < 0.5:
                raise AssertionError(f"{row['name']} tip displacement out of range: {row['disp']}")

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
