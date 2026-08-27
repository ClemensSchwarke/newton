# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Watch The Unified Block Against The Split One
#
# Two identical free-floating beams, same mass, same coupled modal basis, same
# deflection released at t = 0, nothing holding or driving either. The only
# difference is how the solver arranges the reduced elastic system:
#
#   near beam (green frame)  unified: the floating frame and the modal
#                            coordinate are solved together in one (6 + n_m)
#                            block, so the frame recoils against the bulge and
#                            the body center of mass stays pinned.
#   far beam (red frame)     split: the rigid solve owns the frame, the
#                            elastic block owns only the mode, and the two
#                            trade information through a Gauss-Seidel sweep.
#                            The exchange is lossy and the center of mass
#                            walks away.
#
# The white line is each beam's center of mass. It should not move. Watching
# the two lines is the whole demonstration.
#
# One solver cannot hold both arrangements at once, so each beam is stepped in
# its own single-body model and the poses are copied into a display model that
# holds both.
#
# Command:
#   python -m me.scripts.view_elastic_block_comparison
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._reduced_elastic import (
    FRAME_AXES,
    beam_render_sample_points,
    quat_rotate,
    set_camera_from_bounds,
)

BEAM_LENGTH = 1.0
BEAM_HALF_Y = 0.05
BEAM_HALF_Z = 0.04
BEAM_MASS = 1.0
BEAM_DEFLECTION = 0.12
BEAM_MODE_STIFFNESS = 10.0

HEIGHT = 1.0
Y_GAP = 0.6
SUBSTEPS = 8
ITERATIONS = 4
FPS = 60

ARRANGEMENTS = (("unified", True, -0.5 * Y_GAP), ("split", False, 0.5 * Y_GAP))


def build_basis():
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
        label="comparison_basis",
    )


def add_beam(builder, basis, y):
    shape_cfg = newton.ModelBuilder.ShapeConfig()
    shape_cfg.density = 0.0
    shape_cfg.has_shape_collision = False
    shape_cfg.has_particle_collision = False
    beam = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.0, y, HEIGHT), wp.quat_identity()),
        com=wp.vec3(0.0, 0.0, 0.0),
        mass=BEAM_MASS,
        inertia=wp.mat33(0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02),
        mode_q=[BEAM_DEFLECTION],
        modal_basis=basis,
    )
    builder.add_shape_box(
        beam,
        xform=wp.transform_identity(),
        hx=0.5 * BEAM_LENGTH,
        hy=BEAM_HALF_Y,
        hz=BEAM_HALF_Z,
        cfg=shape_cfg,
    )
    return beam


def owner_q_start(model, body):
    elastic_index = int(model.body_elastic_index.numpy()[body])
    owner = int(model.elastic_joint.numpy()[elastic_index])
    return int(model.joint_q_start.numpy()[owner])


class Example:
    def __init__(self, viewer, args):
        self.fps = FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.args = args

        basis = build_basis()
        self.com_factor = float(basis.mode_coupling_linear[0][2]) / BEAM_MASS

        display_builder = newton.ModelBuilder(gravity=0.0)
        self.display_body = {}
        for name, _unified, y in ARRANGEMENTS:
            self.display_body[name] = add_beam(display_builder, basis, y)
        display_builder.color()
        self.model = display_builder.finalize()
        self.display_state = self.model.state()
        self.display_q_start = {name: owner_q_start(self.model, body) for name, body in self.display_body.items()}

        self.sims = {}
        for name, unified, y in ARRANGEMENTS:
            builder = newton.ModelBuilder(gravity=0.0)
            body = add_beam(builder, basis, y)
            builder.color()
            model = builder.finalize()
            solver = newton.solvers.SolverVBD(model, iterations=ITERATIONS)
            solver._elastic_frame_in_block = unified
            self.sims[name] = {
                "model": model,
                "solver": solver,
                "state_0": model.state(),
                "state_1": model.state(),
                "control": model.control(),
                "body": body,
                "q_start": owner_q_start(model, body),
            }

        self.com0 = {name: self._com(name) for name in self.sims}
        self.max_com_drift = dict.fromkeys(self.sims, 0.0)
        self.max_frame_move = dict.fromkeys(self.sims, 0.0)
        self.frame0 = {name: self._frame_z(name) for name in self.sims}

        self.viewer.set_model(self.model)
        self.viewer.show_elastic_strain = True
        self.viewer.elastic_strain_color_max = 0.12
        half = 0.5 * BEAM_LENGTH
        y_half = 0.5 * Y_GAP + BEAM_HALF_Y
        set_camera_from_bounds(
            self.viewer,
            np.array([-half - 0.1, -y_half - 0.1, HEIGHT - 0.3]),
            np.array([half + 0.1, y_half + 0.1, HEIGHT + 0.3]),
            np.array([-0.4, -1.0, 0.45]),
        )
        self._sync_display()

    def _frame_z(self, name):
        sim = self.sims[name]
        return float(sim["state_0"].joint_q.numpy()[sim["q_start"] + 2])

    def _mode_value(self, name):
        sim = self.sims[name]
        return float(sim["state_0"].joint_q.numpy()[sim["q_start"] + 7])

    def _com(self, name):
        return self._frame_z(name) + self.com_factor * self._mode_value(name)

    def _sync_display(self):
        body_q = self.display_state.body_q.numpy()
        joint_q = self.display_state.joint_q.numpy()
        for name, sim in self.sims.items():
            body_q[self.display_body[name]] = sim["state_0"].body_q.numpy()[sim["body"]]
            source = sim["state_0"].joint_q.numpy()
            start = self.display_q_start[name]
            joint_q[start : start + 8] = source[sim["q_start"] : sim["q_start"] + 8]
        self.display_state.body_q.assign(body_q)
        self.display_state.joint_q.assign(joint_q)

    def simulate(self):
        for _ in range(self.sim_substeps):
            for sim in self.sims.values():
                sim["state_0"].clear_forces()
                sim["solver"].step(sim["state_0"], sim["state_1"], sim["control"], None, self.sim_dt)
                sim["state_0"], sim["state_1"] = sim["state_1"], sim["state_0"]

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        for name in self.sims:
            self.max_com_drift[name] = max(self.max_com_drift[name], abs(self._com(name) - self.com0[name]))
            self.max_frame_move[name] = max(self.max_frame_move[name], abs(self._frame_z(name) - self.frame0[name]))
        self._sync_display()

    def _log_overlays(self):
        axis_len = 0.225
        starts, ends, colors = [], [], []
        frame_color = {"unified": np.array([0.1, 0.9, 0.2]), "split": np.array([0.95, 0.2, 0.1])}
        for name, sim in self.sims.items():
            bq = sim["state_0"].body_q.numpy()[sim["body"]]
            origin = np.array(bq[:3], dtype=np.float32)
            for axis, _color in FRAME_AXES:
                starts.append(origin)
                ends.append((origin + quat_rotate(bq[3:7], axis) * axis_len).astype(np.float32))
                colors.append(frame_color[name].astype(np.float32))
            y = origin[1]
            com_z = self.com0[name] + (self._com(name) - self.com0[name])
            starts.append(np.array([-0.75 * BEAM_LENGTH, y, com_z], dtype=np.float32))
            ends.append(np.array([0.75 * BEAM_LENGTH, y, com_z], dtype=np.float32))
            colors.append(np.array([1.0, 1.0, 1.0], dtype=np.float32))
        self.viewer.log_lines(
            "overlays",
            wp.array(np.array(starts, dtype=np.float32), dtype=wp.vec3),
            wp.array(np.array(ends, dtype=np.float32), dtype=wp.vec3),
            wp.array(np.array(colors, dtype=np.float32), dtype=wp.vec3),
            width=0.012,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.display_state)
        self._log_overlays()
        self.viewer.end_frame()

    def test_final(self):
        for name, sim in self.sims.items():
            if not np.isfinite(sim["state_0"].joint_q.numpy()).all():
                raise AssertionError(f"{name} joint coordinates contain non-finite values")
        if self.max_com_drift["unified"] > 0.1 * self.max_com_drift["split"]:
            raise AssertionError(
                f"unified center of mass drifted {self.max_com_drift['unified']:.3e} "
                f"against split {self.max_com_drift['split']:.3e}"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
