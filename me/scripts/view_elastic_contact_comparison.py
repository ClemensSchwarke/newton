# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Watch The Unified Block Against The Split One, Under Contact
#
# The companion to view_elastic_block_comparison, which shows the same two
# arrangements on a free-floating beam. Here each elastic body is a flat slab
# given a sideways push, sliding to a stop on frictional ground. Its single
# mode is a lateral shear running from one at the contacting bottom face to
# zero at the top, so friction acts in the tangent plane on a surface the mode
# moves. This is the case the unified block exists for.
#
#   near slab (green frame)  unified: frame and modal coordinate solved
#                            together in one (6 + n_m) block.
#   far slab (red frame)     split: the rigid solve owns the frame, the
#                            elastic block owns the mode, and the two trade
#                            through a Gauss-Seidel sweep.
#
# Friction drags the contacting face backwards relative to the body, so the
# mode should shear negative and the slab should finish forward of where it
# started. Both arrangements agree on that once converged.
#
# At ITERATIONS = 16 the unified slab is converged, within 2% of its
# 256-iteration answer. The split slab is not: it finishes *behind* its start
# with the shear reversed. Raise ITERATIONS to 64 and the two land on top of
# each other. That is the whole comparison -- they agree on the answer and
# differ in how many iterations reaching it takes.
#
# The slab is deliberately flat. An earlier version used a cube, which tips
# over while sliding; tumbling is chaotic and compares nothing.
#
# One solver holds one arrangement, so each pair is stepped in its own model
# and the poses are copied into a display model holding both.
#
# Command:
#   python -m me.scripts.view_elastic_contact_comparison
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._reduced_elastic import FRAME_AXES, box_surface_mesh, quat_rotate, set_camera_from_bounds

SLAB_HX = 0.12
SLAB_HY = 0.12
SLAB_HZ = 0.02
SLAB_MASS = 1.0
SLAB_MODE_STIFFNESS = 120.0
SLAB_DEFLECTION = 0.0
SLAB_MODE_DAMPING = 1.0
START_HEIGHT = 0.0198
PUSH_SPEED = 0.6

Y_GAP = 0.55
SUBSTEPS = 8
ITERATIONS = 16
ELASTIC_CONTACT_RELAXATION = None
CONTACT_KE = 2.0e4
FPS = 60

ARRANGEMENTS = (("unified", True, -0.5 * Y_GAP), ("split", False, 0.5 * Y_GAP))


def build_basis():
    points, _faces = box_surface_mesh(2.0 * SLAB_HX, SLAB_HY, SLAB_HZ)
    phi = np.zeros_like(points, dtype=np.float32)
    phi[:, 0] = 0.5 - points[:, 2] / (2.0 * SLAB_HZ)
    sample_mass = np.full(points.shape[0], SLAB_MASS / points.shape[0], dtype=np.float32)
    return newton.ModalBasis(
        sample_points=points,
        sample_phi=phi.reshape((-1, 1, 3)),
        sample_mass=sample_mass,
        mode_stiffness=[SLAB_MODE_STIFFNESS],
        mode_damping=[SLAB_MODE_DAMPING],
        label="contact_comparison_basis",
    )


def contact_cfg():
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.ke = CONTACT_KE
    cfg.kd = 30.0
    cfg.mu = 0.5
    cfg.margin = 0.0
    cfg.gap = 0.0
    return cfg


def add_pair(builder, basis, y):
    cfg = contact_cfg()
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.0, y, START_HEIGHT), wp.quat_identity()),
        mass=SLAB_MASS,
        inertia=np.diag([0.004, 0.004, 0.008]).astype(np.float32),
        mode_q=[SLAB_DEFLECTION],
        modal_basis=basis,
    )
    builder.add_shape_box(body, hx=SLAB_HX, hy=SLAB_HY, hz=SLAB_HZ, cfg=cfg)
    return body, body


def owner_joint(model, body):
    elastic_index = int(model.body_elastic_index.numpy()[body])
    return int(model.elastic_joint.numpy()[elastic_index])


def push_elastic_frame(model, state, body, speed):
    """Set the frame's linear velocity through the owner joint.

    ``ModelBuilder.body_qd`` does not reach a reduced elastic body: its floating frame lives
    on the owner joint, so the solver integrates ``joint_qd`` and never reads ``body_qd``.
    The owner joint's velocity is ordered linear first, angular second.
    """
    qd_start = int(model.joint_qd_start.numpy()[owner_joint(model, body)])
    joint_qd = state.joint_qd.numpy()
    joint_qd[qd_start + 0] = speed
    state.joint_qd.assign(joint_qd)


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

        display_builder = newton.ModelBuilder(gravity=-9.81, up_axis="Z")
        display_builder.add_ground_plane(cfg=contact_cfg())
        self.display_bodies = {}
        for name, _unified, y in ARRANGEMENTS:
            self.display_bodies[name] = add_pair(display_builder, basis, y)
        display_builder.color()
        self.model = display_builder.finalize()
        self.display_state = self.model.state()
        self.display_q_start = {
            name: owner_q_start(self.model, bodies[1]) for name, bodies in self.display_bodies.items()
        }

        self.sims = {}
        for name, unified, y in ARRANGEMENTS:
            builder = newton.ModelBuilder(gravity=-9.81, up_axis="Z")
            builder.add_ground_plane(cfg=contact_cfg())
            carrier, body = add_pair(builder, basis, y)
            builder.color()
            model = builder.finalize()
            solver_kwargs = {"rigid_contact_k_start": CONTACT_KE}
            if ELASTIC_CONTACT_RELAXATION is not None:
                solver_kwargs["elastic_contact_relaxation"] = ELASTIC_CONTACT_RELAXATION
            solver = newton.solvers.SolverVBD(model, iterations=ITERATIONS, **solver_kwargs)
            solver._elastic_frame_in_block = unified
            self.sims[name] = {
                "model": model,
                "solver": solver,
                "state_0": model.state(),
                "state_1": model.state(),
                "control": model.control(),
                "contacts": model.contacts(),
                "carrier": carrier,
                "body": body,
                "q_start": owner_q_start(model, body),
            }
            push_elastic_frame(model, self.sims[name]["state_0"], body, PUSH_SPEED)

        self.viewer.set_model(self.model)
        self.viewer.show_elastic_strain = True
        self.viewer.elastic_strain_color_max = 0.04
        set_camera_from_bounds(
            self.viewer,
            np.array([-0.35, -0.5 * Y_GAP - 0.15, -0.02]),
            np.array([0.55, 0.5 * Y_GAP + 0.15, 0.32]),
            np.array([-0.5, -1.0, 0.4]),
        )
        self._sync_display()

    def _sync_display(self):
        body_q = self.display_state.body_q.numpy()
        joint_q = self.display_state.joint_q.numpy()
        for name, sim in self.sims.items():
            carrier, body = self.display_bodies[name]
            source_q = sim["state_0"].body_q.numpy()
            body_q[carrier] = source_q[sim["carrier"]]
            body_q[body] = source_q[sim["body"]]
            source_joint = sim["state_0"].joint_q.numpy()
            start = self.display_q_start[name]
            joint_q[start : start + 8] = source_joint[sim["q_start"] : sim["q_start"] + 8]
        self.display_state.body_q.assign(body_q)
        self.display_state.joint_q.assign(joint_q)

    def simulate(self):
        for _ in range(self.sim_substeps):
            for sim in self.sims.values():
                sim["model"].collide(sim["state_0"], sim["contacts"])
                sim["state_0"].clear_forces()
                sim["solver"].step(sim["state_0"], sim["state_1"], sim["control"], sim["contacts"], self.sim_dt)
                sim["state_0"], sim["state_1"] = sim["state_1"], sim["state_0"]

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self._sync_display()

    def _log_overlays(self):
        axis_len = 0.09
        starts, ends, colors = [], [], []
        frame_color = {"unified": np.array([0.1, 0.9, 0.2]), "split": np.array([0.95, 0.2, 0.1])}
        for name, sim in self.sims.items():
            bq = sim["state_0"].body_q.numpy()[sim["body"]]
            origin = np.array(bq[:3], dtype=np.float32)
            for axis, _color in FRAME_AXES:
                starts.append(origin)
                ends.append((origin + quat_rotate(bq[3:7], axis) * axis_len).astype(np.float32))
                colors.append(frame_color[name].astype(np.float32))
        self.viewer.log_lines(
            "overlays",
            wp.array(np.array(starts, dtype=np.float32), dtype=wp.vec3),
            wp.array(np.array(ends, dtype=np.float32), dtype=wp.vec3),
            wp.array(np.array(colors, dtype=np.float32), dtype=wp.vec3),
            width=0.005,
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


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
