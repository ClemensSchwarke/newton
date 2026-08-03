# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Basic Reduced Elastic Blade Drop
#
# Drops two identical leaf-spring blades onto the ground and varies only the
# contact damping coefficient kd. Each blade is a reduced elastic body built
# from a curved-beam modal basis, clamped at its root to a rigid payload mass
# that is constrained to a vertical prismatic joint, so the only motion is the
# drop, the flex of the blade, and the rebound.
#
# A penalty contact contributes a rank-one damping matrix kd * psi psi^T to the
# modal coordinates, where psi_i = phi_i(p_c) . n projects each mode shape at
# the contact point onto the contact normal. The added modal damping ratio is
#
#     zeta_i = kd * psi_i^2 / (2 * omega_i)
#
# which is only active while the contact is. The left blade uses a kd sized by
# that rule (contact damping <= the material damping the basis was built with);
# the right blade uses a kd inherited from a rigid-foot preset, which overdamps
# the fundamental mode by orders of magnitude. The left blade stores the impact
# energy and returns it. The right blade behaves like putty: it barely deflects,
# dissipates the impact in the contact dashpot, and does not rebound.
#
# Command: python -m newton.examples basic_reduced_elastic_blade_drop
#
###########################################################################

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import Axis

# Traced ANYmal-D blade centerline from the compliance repo. Override the repo
# location with COMPLIANCE_ROOT, or pass --centerline directly. When the file is
# not present the example falls back to a synthetic curved cantilever.
COMPLIANCE_ROOT = Path(os.environ.get("COMPLIANCE_ROOT", Path.home() / "git" / "isaac" / "compliance"))
DEFAULT_CENTERLINE = (
    COMPLIANCE_ROOT
    / "source"
    / "compliance"
    / "compliance"
    / "assets"
    / "data"
    / "Anymal-D-Compliant"
    / "xcel_short_exported_centerline.npy"
)

# Blade material and geometry (ANYmal-D compliant shank leaf spring).
BLADE_WIDTH = 0.06
BLADE_DEPTH = 0.01
BLADE_DENSITY = 1600.0
# Deliberately an order of magnitude below the real blade's 6e10 Pa. At the
# stiff value the blade carries a 10 kg payload on a few millimetres of flex,
# less than the penalty contact's own penetration, so the modal response is
# swamped. Softening by 10x drops the fundamental to ~6.6 Hz and puts the
# static sag in the centimetre range, where the mode bending is the dominant
# compliance in the scene. Pass --young-modulus 6e10 for the real blade.
BLADE_YOUNG_MODULUS = 6e9
BLADE_DAMPING_RATIO = 0.01
BLADE_MODE_COUNT = 4

# Contact material. Only kd differs between the two legs.
CONTACT_KE = 8.0e4
CONTACT_MU = 0.75
KD_TUNED = 0.35
KD_RIGID_PRESET = 1.0e4

PAYLOAD_MASS = 10.0
DROP_HEIGHT = 0.08
LEG_SPACING = 0.9

# The blade root clamp must be stiff. If the fixed joint is soft the floating
# frame simply rotates under the contact reaction and the modes never deflect,
# so the blade behaves like a rigid link on a soft mount.
CLAMP_PENALTY_KE = 1.0e8

SUBSTEPS = 32
ITERATIONS = 64
RIGID_CONTACT_MAX = 4096


def _synthetic_centerline(node_count: int = 26, length: float = 0.75) -> np.ndarray:
    """A curved cantilever standing in for the traced blade centerline.

    Tangent angle grows as ``(s / L)^6``, so the beam runs almost straight down
    from the root and curls forward only near the tip, like the leaf spring it
    approximates. Returned in the ``yz`` plane with the root at the origin.
    """
    s = np.linspace(0.0, length, node_count)
    ds = s[1] - s[0]
    theta = 3.2 * (s / length) ** 6
    y = np.concatenate([[0.0], np.cumsum(np.sin(theta[1:]) * ds)])
    z = -np.concatenate([[0.0], np.cumsum(np.cos(theta[1:]) * ds)])
    return np.column_stack([y, z])


def _contact_config(kd: float) -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.ke = CONTACT_KE
    cfg.kd = kd
    cfg.mu = CONTACT_MU
    cfg.margin = 0.0
    cfg.gap = 0.0
    return cfg


def _visual_config() -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.has_shape_collision = False
    cfg.has_particle_collision = False
    cfg.density = 0.0
    return cfg


def _modal_contact_gains(basis, normal=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Worst-case psi_i^2 over the basis samples for a given contact normal.

    ``psi_i = phi_i(p_c) . n`` is how much of the contact normal mode ``i`` can
    see. The largest value over the surface bounds the damping any single
    contact can inject into that mode.
    """
    phi = np.asarray(basis.sample_phi, dtype=np.float64)  # [samples, modes, 3]
    projected = phi @ np.asarray(normal, dtype=np.float64)
    return np.max(projected**2, axis=0)


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.args = args

        path = Path(args.centerline) if args.centerline else DEFAULT_CENTERLINE
        if path.is_file():
            centerline = np.load(path)
            if centerline.ndim != 2 or centerline.shape[1] != 2:
                raise ValueError(f"centerline must have shape [points, 2], got {centerline.shape}")
            self.centerline_source = str(path)
        elif args.centerline:
            raise FileNotFoundError(f"centerline not found: {path}")
        else:
            centerline = _synthetic_centerline()
            self.centerline_source = f"synthetic (traced blade not found at {path})"

        self.legs = [("tuned", KD_TUNED), ("rigid_preset", KD_RIGID_PRESET)]

        builder = newton.ModelBuilder(up_axis=Axis.Z)
        # Each blade surface carries 4 * node_count elastic samples that can all
        # generate contacts, on top of the ordinary mesh narrow phase.
        builder.num_rigid_contacts_per_world = RIGID_CONTACT_MAX
        visual_cfg = _visual_config()

        self.carriage_bodies = []
        self.blade_bodies = []
        self.blade_shapes = []
        self.mode_stiffness = None
        self.frequencies = None
        self.psi_squared = None

        # One shared, infinite ground plane. A finite add_shape_plane does not
        # reach create_elastic_shape_contacts, so every contact comes back with
        # elastic_sample = -1 and the modal coordinates are bypassed entirely.
        # add_ground_plane does reach it, and produces the fewest contacts of
        # any ground geometry tried.
        ground_cfg = _contact_config(0.0)
        self.ground_shape = builder.add_ground_plane(cfg=ground_cfg)

        for leg_index, (label, kd) in enumerate(self.legs):
            x = (leg_index - 0.5) * LEG_SPACING
            # Contact material is the arithmetic mean of the two shapes. The
            # ground carries kd = 0, so each blade carries 2 * kd to make the
            # pair average come out at the value this leg is meant to test.
            # ke and mu are equal on both sides and pass through unchanged.
            contact_cfg = _contact_config(2.0 * kd)

            generator = newton.ModalGeneratorCurvedBeam(
                centerline=centerline,
                width=BLADE_WIDTH,
                depth=BLADE_DEPTH,
                plane="yz",
                mode_count=BLADE_MODE_COUNT,
                density=BLADE_DENSITY,
                young_modulus=args.young_modulus,
                damping_ratio=BLADE_DAMPING_RATIO,
                label=f"blade_basis_{label}",
            )
            basis = generator.build()
            if self.frequencies is None:
                self.frequencies = np.asarray(generator.frequencies, dtype=np.float64)
                self.mode_stiffness = np.asarray(basis.mode_stiffness, dtype=np.float64)
                self.psi_squared = _modal_contact_gains(basis)
                # Lowest point of the undeformed blade, used to set the drop gap.
                # sample_points already include the cross-section corners.
                self.blade_bottom = float(np.min(np.asarray(basis.sample_points, dtype=np.float64)[:, 2]))

            carriage_z = -self.blade_bottom + DROP_HEIGHT
            carriage_pose = wp.transform(wp.vec3(x, 0.0, carriage_z), wp.quat_identity())

            # add_link, not add_body: add_body would also attach a free joint to
            # the world, leaving the payload with six free DoFs fighting the
            # prismatic penalty constraint.
            carriage = builder.add_link(
                xform=carriage_pose,
                mass=PAYLOAD_MASS,
                inertia=wp.mat33(0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05),
                label=f"payload_{label}",
            )
            builder.add_shape_box(carriage, hx=0.05, hy=0.05, hz=0.05, cfg=visual_cfg)
            slider = builder.add_joint_prismatic(
                parent=-1,
                child=carriage,
                parent_xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                axis=Axis.Z,
                label=f"slider_{label}",
            )
            builder.add_articulation([slider], label=f"slider_articulation_{label}")
            builder.joint_q[builder.joint_q_start[slider]] = carriage_z
            self.carriage_bodies.append(carriage)

            blade = builder.add_body_elastic(
                xform=carriage_pose,
                mass=generator.total_mass,
                com=wp.vec3(*generator.center_of_mass),
                inertia=wp.mat33(*generator.inertia.flatten()),
                modal_basis=basis,
                lock_inertia=True,
                label=f"blade_{label}",
            )
            self.blade_bodies.append(blade)
            builder.add_joint_fixed(parent=carriage, child=blade, label=f"clamp_{label}")

            vertices, indices = generator.surface_mesh()
            blade_shape = builder.add_shape_mesh(
                blade,
                mesh=newton.Mesh(vertices, indices, compute_inertia=False),
                cfg=contact_cfg,
                label=f"blade_mesh_{label}",
            )
            self.blade_shapes.append(blade_shape)

        builder.color()
        self.model = builder.finalize()
        self.model.rigid_contact_max = RIGID_CONTACT_MAX

        fk_state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, fk_state)
        wp.copy(self.model.body_q, fk_state.body_q)
        wp.copy(self.model.body_qd, fk_state.body_qd)

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=ITERATIONS,
            rigid_joint_adaptive_stiffness=False,
            rigid_joint_linear_ke=CLAMP_PENALTY_KE,
            rigid_joint_angular_ke=CLAMP_PENALTY_KE,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        elastic_index = self.model.body_elastic_index.numpy()
        elastic_joint = self.model.elastic_joint.numpy()
        joint_q_start = self.model.joint_q_start.numpy()
        self.modal_q_starts = [
            int(joint_q_start[int(elastic_joint[int(elastic_index[b])])]) + 7 for b in self.blade_bodies
        ]

        self.trace = {"t": [], "z": [[], []], "q1": [[], []], "energy": [[], []]}
        self._report_setup()

        self.viewer.set_model(self.model)
        # Colour the blade surface by strain so the bending is visible.
        self.viewer.show_elastic_strain = True
        self.viewer.set_camera(pos=wp.vec3(0.0, -2.2, 0.6), pitch=-8.0, yaw=90.0)

    def _report_setup(self):
        if self.args.quiet:
            return
        omega = 2.0 * np.pi * self.frequencies
        material_c = 2.0 * BLADE_DAMPING_RATIO * omega
        kd_bound = float(np.min(material_c / self.psi_squared))
        print(f"[blade_drop] centerline: {self.centerline_source}")
        print(f"[blade_drop] young modulus: {self.args.young_modulus:.3g} Pa, payload: {PAYLOAD_MASS} kg")
        print("[blade_drop] blade modes")
        print(f"{'mode':>5} {'f [Hz]':>10} {'omega':>10} {'psi^2':>10} {'c_mat':>10}")
        for i, f in enumerate(self.frequencies):
            print(f"{i + 1:>5} {f:>10.2f} {omega[i]:>10.1f} {self.psi_squared[i]:>10.3f} {material_c[i]:>10.2f}")
        print(f"[blade_drop] design rule kd <= min_i c_i / psi_i^2 = {kd_bound:.3f}")
        for label, kd in self.legs:
            zeta = kd * self.psi_squared / (2.0 * omega)
            print(f"[blade_drop] {label:>13}: kd = {kd:<10.4g} zeta_contact = {np.round(zeta, 4).tolist()}")

    def _sample(self):
        joint_q = self.state_0.joint_q.numpy()
        body_q = self.state_0.body_q.numpy()
        self.trace["t"].append(self.sim_time)
        for i, carriage in enumerate(self.carriage_bodies):
            start = self.modal_q_starts[i]
            modal_q = joint_q[start : start + BLADE_MODE_COUNT]
            self.trace["z"][i].append(float(body_q[carriage, 2]))
            self.trace["q1"][i].append(float(modal_q[0]))
            self.trace["energy"][i].append(0.5 * float(np.sum(self.mode_stiffness * modal_q**2)))

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self._sample()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def summary(self):
        stats = []
        for i, (label, kd) in enumerate(self.legs):
            z = np.asarray(self.trace["z"][i])
            energy = np.asarray(self.trace["energy"][i])
            q1 = np.asarray(self.trace["q1"][i])
            bottom = int(np.argmin(z))
            rebound = float(np.max(z[bottom:]) - z[bottom]) if bottom < len(z) - 1 else 0.0
            stats.append(
                {
                    "label": label,
                    "kd": kd,
                    "peak_q1": float(np.max(np.abs(q1))),
                    "peak_energy": float(np.max(energy)),
                    "lowest_z": float(z[bottom]),
                    "rebound": rebound,
                    "final_q1": float(q1[-1]),
                }
            )
        return stats

    def print_summary(self):
        print(f"\n{'leg':>13} {'kd':>10} {'peak |q1|':>10} {'peak E [J]':>11} {'min z [m]':>10} {'rebound [m]':>12}")
        for s in self.summary():
            print(
                f"{s['label']:>13} {s['kd']:>10.4g} {s['peak_q1']:>10.4f} "
                f"{s['peak_energy']:>11.3f} {s['lowest_z']:>10.4f} {s['rebound']:>12.4f}"
            )

    def plot(self, path: str):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.asarray(self.trace["t"])
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        for i, (label, kd) in enumerate(self.legs):
            tag = f"{label} (kd={kd:g})"
            axes[0].plot(t, self.trace["z"][i], label=tag)
            axes[1].plot(t, self.trace["q1"][i], label=tag)
            axes[2].plot(t, self.trace["energy"][i], label=tag)
        axes[0].set_ylabel("payload height [m]")
        axes[1].set_ylabel("modal coordinate $q_1$")
        axes[2].set_ylabel("blade strain energy [J]")
        axes[2].set_xlabel("time [s]")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        print(f"[blade_drop] wrote {path}")

    def test_final(self):
        stats = self.summary()
        tuned, preset = stats[0], stats[1]
        if tuned["peak_energy"] <= 0.0:
            raise AssertionError("tuned blade stored no strain energy")
        if tuned["peak_q1"] < 4.0 * preset["peak_q1"]:
            raise AssertionError(
                "expected the tuned blade to deflect much further than the overdamped one: "
                f"{tuned['peak_q1']} vs {preset['peak_q1']}"
            )
        if tuned["rebound"] <= preset["rebound"]:
            raise AssertionError(
                f"expected the tuned blade to rebound higher: {tuned['rebound']} vs {preset['rebound']}"
            )
        for i in range(len(self.legs)):
            if not np.all(np.isfinite(self.trace["z"][i])):
                raise AssertionError("payload height contains non-finite values")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--centerline",
            type=str,
            default=None,
            help="Blade centerline .npy of shape [points, 2] in the yz plane. Defaults to the "
            f"traced ANYmal-D blade at {DEFAULT_CENTERLINE}, and to a synthetic curved "
            "cantilever if that file is missing.",
        )
        parser.add_argument(
            "--young-modulus",
            type=float,
            default=BLADE_YOUNG_MODULUS,
            help="Blade Young's modulus [Pa]. The default is softened 10x from the real blade's "
            "6e10 so the modal deflection dominates the contact penetration.",
        )
        parser.add_argument(
            "--plot",
            type=str,
            default=None,
            help="Write a PNG of payload height, modal coordinate and strain energy to this path.",
        )
        return parser


def test(device=None, frame_count: int = 120):
    from newton.examples.basic._reduced_elastic_contact import run_example_test

    return run_example_test(Example, frame_count, device)


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
    example.print_summary()
    if args.plot:
        example.plot(args.plot)
