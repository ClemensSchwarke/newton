#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Interactive viewer for comparing VBD local vs sparse articulation solve.

Usage:
    python reports/vbd_sparse_articulation/view_humanoid.py
    python reports/vbd_sparse_articulation/view_humanoid.py --robot h1 --mode block_sparse_joints --iterations 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import warp as wp

import newton
import newton.examples
from bench_vbd_humanoid_contact import build_humanoid, solver_stiffness_kwargs


_RENDER_FPS = 120
_SUBSTEPS = {"h1": 2, "g1": 3}


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = _RENDER_FPS
        self.sim_time = 0.0
        self.sim_substeps = _SUBSTEPS[args.robot]
        self.sim_dt = 1.0 / (_RENDER_FPS * self.sim_substeps)
        self._slowdown = args.slowdown
        self._phys_budget = 0

        self.model = build_humanoid(args.robot, add_ground=True)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=args.iterations,
            rigid_articulation_solve=args.mode,
            rigid_body_contact_buffer_size=512,
            **solver_stiffness_kwargs(args.joint_stiffness),
        )

        self.viewer.set_model(self.model)
        self._setup_camera()

    def _setup_camera(self):
        body_pos = self.state_0.body_q.numpy()[:, 0:3]
        center = 0.5 * (body_pos.min(axis=0) + body_pos.max(axis=0))
        extent = max(float((body_pos.max(axis=0) - body_pos.min(axis=0)).max()), 1.0)
        target = center + np.array([0.0, 0.0, 0.1])
        eye = target + np.array([1.2 * extent, -1.2 * extent, 0.5 * extent])
        front = target - eye
        front /= np.linalg.norm(front)
        yaw = float(np.degrees(np.arctan2(front[1], front[0])))
        pitch = float(np.degrees(np.arcsin(front[2])))
        self.viewer.set_camera(pos=wp.vec3(*eye.tolist()), pitch=pitch, yaw=yaw)

    def step(self):
        self._phys_budget += 1
        if self._phys_budget < self._slowdown:
            return
        self._phys_budget = 0

        self.model.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.sim_substeps * self.sim_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(render_fps=_RENDER_FPS)
        parser.add_argument("--robot", default="g1", choices=["h1", "g1"])
        parser.add_argument("--mode", default="local", choices=["local", "block_sparse_joints"])
        parser.add_argument("--iterations", type=int, default=3)
        parser.add_argument("--joint-stiffness", default="fixed_high", choices=["default", "fixed_high"])
        parser.add_argument("--slowdown", type=int, default=8, metavar="N", help="run physics every Nth render frame")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
