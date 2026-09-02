# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""State-locked cost of the reduced elastic solve, one run per git worktree.

The free-running benchmark in ``anymal_elastic_arrangement_cost`` lets the two trees reach
different states, so their contact sets differ and the contact kernels are not comparable.
Here one tree dumps a loaded state after ``SETTLE_FRAMES``, both trees restore it before every
timed frame, and the contact count is identical by construction. Set ``MODE = "dump"`` on the
pre-merge tree once, then ``MODE = "time"`` on each tree. The printed FK checksum must match
across trees, otherwise the joint layouts differ and the transplant is invalid.
"""

import collections
import json
import time
import numpy as np
import warp as wp
import newton
from newton.solvers import SolverVBD
import me.scripts.anymal_elastic_arrangement_cost as bench

TREE_TAG = "head"
MODE = "time"
WORLD_COUNT = 1024
SNAP = f"/tmp/claude-1000/snap_{WORLD_COUNT}.npz"
SETTLE_FRAMES = 25
WARMUP = 5
TIMED = 20

model = bench.build_model(WORLD_COUNT)
solver = SolverVBD(
    model,
    iterations=bench.ITERATIONS,
    rigid_joint_adaptive_stiffness=True,
    rigid_joint_linear_ke=bench.JOINT_KE,
    rigid_joint_angular_ke=bench.JOINT_KE,
    rigid_articulation_solve=bench.ARTICULATION_SOLVE,
    rigid_articulation_relaxation=bench.ARTICULATION_RELAXATION,
)
s0, s1 = model.state(), model.state()
control, contacts = model.control(), model.contacts()
dt = bench.FRAME_DT / bench.SUBSTEPS

FIELDS = ("body_q", "body_qd", "joint_q", "joint_qd")


def frame():
    global s0, s1
    model.collide(s0, contacts)
    for _ in range(bench.SUBSTEPS):
        s0.clear_forces()
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0


if MODE == "dump":
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    for _ in range(SETTLE_FRAMES):
        frame()
    np.savez(SNAP, **{f: getattr(s0, f).numpy() for f in FIELDS})
    print(f"[{TREE_TAG}] dumped shapes " + " ".join(f"{f}{getattr(s0, f).numpy().shape}" for f in FIELDS), flush=True)
    raise SystemExit(0)

snap = np.load(SNAP)
for f in FIELDS:
    assert snap[f].shape == getattr(s0, f).numpy().shape, (f, snap[f].shape, getattr(s0, f).numpy().shape)


def restore():
    for f in FIELDS:
        getattr(s0, f).assign(snap[f])


restore()
newton.eval_fk(model, s0.joint_q, s0.joint_qd, s1)
print(f"[{TREE_TAG}] fk checksum {float(np.sum(s1.body_q.numpy())):.6f}", flush=True)

for _ in range(WARMUP):
    restore()
    frame()
wp.synchronize_device()

t0 = time.perf_counter()
for _ in range(TIMED):
    restore()
    frame()
wp.synchronize_device()
wall = (time.perf_counter() - t0) / TIMED * 1e3

per_kernel = collections.defaultdict(float)
with wp.ScopedTimer("k", cuda_filter=wp.TIMING_KERNEL, print=False) as timer:
    for _ in range(TIMED):
        restore()
        frame()
    wp.synchronize_device()
for r in timer.timing_results:
    per_kernel[r.name.replace("forward kernel ", "").split("<")[0]] += r.elapsed
per_kernel = {k: v / TIMED for k, v in per_kernel.items()}
nc = int(contacts.rigid_contact_count.numpy()[0])
print(
    f"[{TREE_TAG}] worlds={WORLD_COUNT} wall {wall:9.3f}  gpu {sum(per_kernel.values()):9.3f}  contacts {nc}",
    flush=True,
)
json.dump(
    {"wall": wall, "gpu": sum(per_kernel.values()), "contacts": nc, "kernels": per_kernel},
    open(f"/tmp/claude-1000/locked_{TREE_TAG}_{WORLD_COUNT}.json", "w"),
    indent=2,
)
