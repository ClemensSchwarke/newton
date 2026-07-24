# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Record Simulation Metrics
#
# Runs a newton example headless (ViewerNull, no GL) and records per-frame
# joint constraint violations, joint angles, and contact penetration for
# parameter tuning. Exports a small JSON summary+series and a time-series plot.
#
# Data recording is fully headless and fast (ViewerNull). Set RECORD_VIDEO =
# True to also render a matching-named video via capture_video.py; that path
# needs a GL context, so run under xvfb when enabling it:
#   xvfb-run -a uv run python -m me.scripts.record_metrics
#
# Edit the config constants below and run:
#   python -m me.scripts.record_metrics
###########################################################################

import importlib
import inspect
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

import newton
import newton.examples
import newton.viewer

EXAMPLE = "robot_anymal_compliant"
EXAMPLE_ARGS = [
    "--usd-path",
    "/home/clem/ETH/PhD/Projects/Compliance/Assets/usd/anymal_d_compliant_svg_xcel_short_exported_n10.usd",
]
NUM_FRAMES = 120
DEVICE = "cuda:0"
OUT_STEM = "robot_anymal_compliant"
OVERRIDES = {"SUBSTEPS": 5, "ITERATIONS": 3, "CONTACT_KE": 8.0e5}
RECORD_VIDEO = True
MEASURE_SPEED = True
SPEED_FRAMES = 100

MOTOR_SUFFIXES = ("_HAA", "_HFE", "_KFE")
SPRING_SUBSTR = "_shank_spring_"

NEWTON_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = NEWTON_ROOT / "me" / "data"


def _resolve_example(name):
    examples = newton.examples.get_examples()
    if name not in examples:
        raise SystemExit(f"unknown example '{name}'. available: {', '.join(sorted(examples))}")
    return importlib.import_module(examples[name])


def _example_config(module):
    return {
        k: v for k, v in vars(module).items() if k.isupper() and isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def _run_stem():
    def tag(v):
        return f"{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)

    return OUT_STEM + "".join(f"_{k.lower()}_{tag(v)}" for k, v in OVERRIDES.items())


def _measure_speed(example, timed_frames):
    substeps = example.sim_substeps
    dt = example.sim_dt
    model, solver, control, contacts = example.model, example.solver, example.control, example.contacts
    s0, s1 = example.state_0, example.state_1
    model.collide(s0, contacts)

    def run_substeps(a, b):
        for _ in range(substeps):
            a.clear_forces()
            solver.step(a, b, control, contacts, dt)
            a, b = b, a
        return a, b

    graph = None
    method = "eager"
    if wp.get_device().is_cuda:
        try:
            with wp.ScopedCapture() as capture:
                run_substeps(s0, s1)
            graph = capture.graph
            method = "cuda_graph"
        except Exception as exc:
            method = f"eager ({type(exc).__name__})"

    wp.synchronize_device()
    start = time.perf_counter()
    if graph is not None:
        for _ in range(timed_frames):
            wp.capture_launch(graph)
    else:
        a, b = s0, s1
        for _ in range(timed_frames):
            a, b = run_substeps(a, b)
    wp.synchronize_device()
    elapsed = time.perf_counter() - start

    return {
        "method": method,
        "note": "solver substeps only (excludes collision)",
        "timed_frames": timed_frames,
        "substeps_per_frame": substeps,
        "solver_ms_per_frame": 1.0e3 * elapsed / timed_frames,
        "solver_ms_per_substep": 1.0e3 * elapsed / (timed_frames * substeps),
        "solver_frames_per_second": timed_frames / elapsed,
    }


def _solver_config(solver):
    cfg = {}
    for name in (
        "iterations",
        "rigid_articulation_solve",
        "rigid_articulation_relaxation",
        "rigid_articulation_diagonal_regularization",
        "rigid_avbd_gamma",
        "rigid_joint_alpha",
        "rigid_contact_hard",
        "rigid_joint_linear_ke",
        "rigid_joint_angular_ke",
        "rigid_joint_linear_kd",
        "rigid_joint_angular_kd",
    ):
        v = getattr(solver, name, None)
        if v is not None:
            cfg[name] = v if isinstance(v, (int, str)) else float(v)
    try:
        kmin = solver.joint_penalty_k_min.numpy()
        kmax = solver.joint_penalty_k_max.numpy()
        cfg["joint_penalty_k_min"] = float(kmin.min())
        cfg["joint_penalty_k_max"] = float(kmax.max())
        cfg["joint_penalty_fixed_k"] = bool(np.allclose(kmin, kmax))
    except Exception:
        pass
    sig = inspect.signature(type(solver))
    for name in ("rigid_avbd_alpha", "rigid_avbd_beta", "rigid_avbd_contact_alpha"):
        if name in sig.parameters:
            cfg.setdefault(f"{name}_default", sig.parameters[name].default)
    return cfg


def _qmul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def _qconj(q):
    out = q.copy()
    out[..., 0:3] = -out[..., 0:3]
    return out


def _qrot(q, v):
    qv = q[..., 0:3]
    t = 2.0 * np.cross(qv, v)
    return v + q[..., 3:4] * t + np.cross(qv, t)


def _rotvec(q):
    q = np.where(q[..., 3:4] < 0.0, -q, q)
    w = np.clip(q[..., 3], -1.0, 1.0)
    s = np.sqrt(np.maximum(0.0, 1.0 - w * w))
    small = s < 1.0e-8
    scale = np.where(small, 2.0, 2.0 * np.arccos(w) / np.where(small, 1.0, s))
    return q[..., 0:3] * scale[..., None]


class _JointMeta:
    def __init__(self, model):
        jt = model.joint_type.numpy()
        parent = model.joint_parent.numpy()
        child = model.joint_child.numpy()
        xp = model.joint_X_p.numpy()
        xc = model.joint_X_c.numpy()
        axis = model.joint_axis.numpy()
        qd_start = model.joint_qd_start.numpy()
        q_start = model.joint_q_start.numpy()
        labels = getattr(model, "joint_key", None)
        if labels is None:
            labels = getattr(model, "joint_label", None)
        labels = list(labels) if labels is not None else [f"joint_{j}" for j in range(model.joint_count)]

        rev = int(newton.JointType.REVOLUTE)
        idx = [j for j in range(model.joint_count) if int(jt[j]) == rev]
        self.joint_idx = idx
        self.parent = np.array([parent[j] for j in idx], dtype=np.int64)
        self.child = np.array([child[j] for j in idx], dtype=np.int64)
        self.xp_p = np.array([xp[j][0:3] for j in idx], dtype=np.float64)
        self.xp_q = np.array([xp[j][3:7] for j in idx], dtype=np.float64)
        self.xc_p = np.array([xc[j][0:3] for j in idx], dtype=np.float64)
        self.xc_q = np.array([xc[j][3:7] for j in idx], dtype=np.float64)
        ax = np.array([axis[qd_start[j]] for j in idx], dtype=np.float64)
        self.axis = ax / np.maximum(np.linalg.norm(ax, axis=1, keepdims=True), 1.0e-12)
        self.q_start = np.array([q_start[j] for j in idx], dtype=np.int64)
        self.names = [str(labels[j]).rsplit("/", 1)[-1] for j in idx]
        self.is_motor = np.array([n.endswith(MOTOR_SUFFIXES) for n in self.names])
        self.is_spring = np.array([SPRING_SUBSTR in n for n in self.names])


def _joint_violations(body_q, meta):
    pos = body_q[:, 0:3]
    quat = body_q[:, 3:7]
    has_parent = meta.parent >= 0
    parent_idx = np.where(has_parent, meta.parent, 0)
    pw_p = np.where(has_parent[:, None], pos[parent_idx] + _qrot(quat[parent_idx], meta.xp_p), meta.xp_p)
    pw_q = np.where(has_parent[:, None], _qmul(quat[parent_idx], meta.xp_q), meta.xp_q)
    cw_p = pos[meta.child] + _qrot(quat[meta.child], meta.xc_p)
    cw_q = _qmul(quat[meta.child], meta.xc_q)

    lin = np.linalg.norm(pw_p - cw_p, axis=1)
    rv = _rotvec(_qmul(_qconj(pw_q), cw_q))
    along = np.sum(rv * meta.axis, axis=1, keepdims=True)
    ang = np.linalg.norm(rv - along * meta.axis, axis=1)
    return lin, ang


def _contact_penetration(contacts, shape_body, body_q):
    try:
        count = int(contacts.rigid_contact_count.numpy()[0])
        if count <= 0:
            return 0.0, 0.0, 0
        s0 = contacts.rigid_contact_shape0.numpy()[:count]
        s1 = contacts.rigid_contact_shape1.numpy()[:count]
        cp0 = contacts.rigid_contact_point0.numpy()[:count].astype(np.float64)
        cp1 = contacts.rigid_contact_point1.numpy()[:count].astype(np.float64)
        n = contacts.rigid_contact_normal.numpy()[:count].astype(np.float64)
        m0 = contacts.rigid_contact_margin0.numpy()[:count].astype(np.float64)
        m1 = contacts.rigid_contact_margin1.numpy()[:count].astype(np.float64)
        pos = body_q[:, 0:3]
        quat = body_q[:, 3:7]

        def to_world(shape, cp):
            body = np.where(shape >= 0, shape_body[np.clip(shape, 0, None)], -1)
            has = body >= 0
            bi = np.where(has, body, 0)
            return np.where(has[:, None], pos[bi] + _qrot(quat[bi], cp), cp)

        sep = np.sum(n * (to_world(s1, cp1) - to_world(s0, cp0)), axis=1) - (m0 + m1)
        pen = np.clip(-sep, 0.0, None)
        positive = pen[pen > 0.0]
        return float(pen.max()), (float(positive.mean()) if positive.size else 0.0), count
    except Exception:
        return 0.0, 0.0, 0


def _plot(series, angles, meta, path, stem):
    t = series["time"]
    fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    for k in range(len(meta.names)):
        if meta.is_spring[k]:
            ax[0].plot(t, angles[:, k], color="0.7", lw=0.6, alpha=0.5)
    for k in range(len(meta.names)):
        if meta.is_motor[k]:
            ax[0].plot(t, angles[:, k], lw=1.2, label=meta.names[k])
    ax[0].set_ylabel("joint angle [rad]")
    ax[0].set_title(f"{stem}: motors (colored) + shank springs (grey)")
    ax[0].legend(fontsize=6, ncol=4, loc="upper right")

    ax[1].plot(t, np.array(series["joint_lin_viol_max"]) * 1.0e3, label="max", color="#a6411d")
    ax[1].plot(t, np.array(series["joint_lin_viol_avg"]) * 1.0e3, label="avg", color="#e0955a")
    ax[1].set_ylabel("joint linear\nviolation [mm]")
    ax[1].legend(fontsize=8)

    ax[2].plot(t, series["joint_ang_viol_max"], label="max", color="#006c67")
    ax[2].plot(t, series["joint_ang_viol_avg"], label="avg", color="#5aa9a5")
    ax[2].set_ylabel("joint angular\nviolation [rad]")
    ax[2].legend(fontsize=8)

    ax[3].plot(t, np.array(series["contact_pen_max"]) * 1.0e3, label="max", color="#333")
    ax[3].plot(t, np.array(series["contact_pen_avg"]) * 1.0e3, label="avg", color="#999")
    ax[3].set_ylabel("contact\npenetration [mm]")
    ax[3].set_xlabel("time [s]")
    ax[3].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    example_module = _resolve_example(EXAMPLE)
    for key, value in OVERRIDES.items():
        setattr(example_module, key, value)
    stem = _run_stem()
    cls = example_module.Example
    parser = cls.create_parser()
    args = parser.parse_args(["--viewer", "null", "--device", DEVICE, *EXAMPLE_ARGS])
    args.headless = True

    with wp.ScopedDevice(DEVICE):
        viewer = newton.viewer.ViewerNull(num_frames=NUM_FRAMES)
        example = cls(viewer, args)
        model = example.model
        meta = _JointMeta(model)
        shape_body = model.shape_body.numpy()
        jq = wp.zeros_like(model.joint_q)
        jqd = wp.zeros_like(model.joint_qd)

        times, angles = [], []
        lin_max, lin_avg, ang_max, ang_avg, pen_max, pen_avg, n_contacts = [], [], [], [], [], [], []

        for frame in range(NUM_FRAMES):
            example.step()
            newton.eval_ik(model, example.state_0, jq, jqd)
            angles.append(jq.numpy()[meta.q_start].copy())

            body_q = example.state_0.body_q.numpy().astype(np.float64)
            lin, ang = _joint_violations(body_q, meta)
            model.collide(example.state_0, example.contacts)
            pmax, pavg, nc = _contact_penetration(example.contacts, shape_body, body_q)

            times.append(frame * example.frame_dt)
            lin_max.append(float(lin.max()))
            lin_avg.append(float(lin.mean()))
            ang_max.append(float(ang.max()))
            ang_avg.append(float(ang.mean()))
            pen_max.append(pmax)
            pen_avg.append(pavg)
            n_contacts.append(nc)

        speed = _measure_speed(example, SPEED_FRAMES) if MEASURE_SPEED else None

    angles_arr = np.array(angles)
    series = {
        "time": times,
        "joint_lin_viol_max": lin_max,
        "joint_lin_viol_avg": lin_avg,
        "joint_ang_viol_max": ang_max,
        "joint_ang_viol_avg": ang_avg,
        "contact_pen_max": pen_max,
        "contact_pen_avg": pen_avg,
        "n_contacts": n_contacts,
        "joint_angles": angles_arr.tolist(),
    }
    payload = {
        "example": EXAMPLE,
        "num_frames": NUM_FRAMES,
        "fps": example.fps,
        "dt": example.frame_dt,
        "device": DEVICE,
        "joint_count": int(model.joint_count),
        "revolute_count": len(meta.names),
        "config": {
            "overrides": OVERRIDES,
            "example": _example_config(example_module),
            "sim": {
                "substeps": example.sim_substeps,
                "fps": example.fps,
                "frame_dt": example.frame_dt,
                "sim_dt": example.sim_dt,
            },
            "solver": _solver_config(example.solver),
        },
        "joint_names": meta.names,
        "joint_is_motor": meta.is_motor.tolist(),
        "joint_is_spring": meta.is_spring.tolist(),
        "speed": speed,
        "summary": {
            "joint_lin_viol_max": max(lin_max),
            "joint_lin_viol_mean": float(np.mean(lin_avg)),
            "joint_lin_viol_final": lin_max[-1],
            "joint_ang_viol_max": max(ang_max),
            "joint_ang_viol_mean": float(np.mean(ang_avg)),
            "joint_ang_viol_final": ang_max[-1],
            "contact_pen_max": max(pen_max),
            "contact_pen_mean": float(np.mean([p for p in pen_avg if p > 0.0]))
            if any(p > 0.0 for p in pen_avg)
            else 0.0,
            "contact_pen_final": pen_max[-1],
        },
        "series": series,
    }

    json_path = OUT_DIR / f"{stem}_metrics.json"
    plot_path = OUT_DIR / f"{stem}_metrics.png"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    _plot(series, angles_arr, meta, plot_path, stem)

    s = payload["summary"]
    print(f"[record] joints={payload['revolute_count']} frames={NUM_FRAMES}")
    print(
        f"[record] joint linear violation  max={s['joint_lin_viol_max']:.3e} m  mean={s['joint_lin_viol_mean']:.3e} m"
    )
    print(
        f"[record] joint angular violation max={s['joint_ang_viol_max']:.3e} rad mean={s['joint_ang_viol_mean']:.3e} rad"
    )
    print(f"[record] contact penetration     max={s['contact_pen_max']:.3e} m  final={s['contact_pen_final']:.3e} m")
    if speed is not None:
        print(
            f"[record] solver speed [{speed['method']}]  {speed['solver_ms_per_frame']:.3f} ms/frame  "
            f"{speed['solver_ms_per_substep']:.4f} ms/substep  {speed['solver_frames_per_second']:.0f} frames/s"
        )
    print(f"[record] wrote {json_path}")
    print(f"[record] wrote {plot_path}")

    if RECORD_VIDEO:
        import me.scripts.capture_video as capture_video  # noqa: PLC0415

        video_path = OUT_DIR / f"{stem}.mp4"
        print(f"[record] capturing video -> {video_path}")
        try:
            with wp.ScopedDevice(DEVICE):
                capture_video.main(
                    [
                        EXAMPLE,
                        "--out",
                        str(video_path),
                        "--num-frames",
                        str(NUM_FRAMES),
                        "--device",
                        DEVICE,
                        *EXAMPLE_ARGS,
                    ]
                )
        except Exception as exc:
            print(
                f"[record] video capture failed ({exc}); metrics were still written. Run under xvfb for a GL context."
            )


if __name__ == "__main__":
    main()
