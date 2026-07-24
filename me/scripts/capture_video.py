# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Capture Example Video
#
# Drives any newton example headless through ViewerGL, grabs each rendered
# frame with viewer.get_frame(), and pipes raw RGB into the system ffmpeg
# encoded at a fixed frame rate. Each example step advances frame_dt = 1/fps
# of sim time and produces exactly one video frame, so the resulting mp4
# plays back at real time no matter how slow the simulation runs.
#
# Command:
#   python -m me.scripts.capture_video robot_anymal_compliant \
#       --out anymal_compliant.mp4 --num-frames 300 \
#       --usd-path /path/to/anymal_d_compliant_svg_xcel_short_exported_n10.usd
#
# Pass any example-specific flags (--usd-path, --centerline, ...) straight
# through; they are parsed by the example's own create_parser().
###########################################################################

import argparse
import importlib
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton.examples
from newton._src.viewer.viewer_gl import ViewerGL

NEWTON_ROOT = Path(__file__).resolve().parents[2]


def _resolve_example(name):
    examples = newton.examples.get_examples()
    if name not in examples:
        options = ", ".join(sorted(examples))
        raise SystemExit(f"unknown example '{name}'. available: {options}")

    module = importlib.import_module(examples[name])
    return module.Example


def _frame_camera(viewer, example, azimuth_deg, elevation_deg, zoom):
    state = getattr(example, "state_0", None)
    if state is None:
        return
    positions = state.body_q.numpy()[:, :3]
    if positions.size == 0:
        return

    center = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    extent = float(np.max(positions.max(axis=0) - positions.min(axis=0)))
    extent = max(extent, 1.0)

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    direction = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    distance = extent * 1.6 * zoom
    pos = center + distance * direction

    viewer.set_camera(wp.vec3(*pos), -elevation_deg, azimuth_deg + 180.0)


def main(argv):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("example")
    pre.add_argument(
        "--out",
        default=None,
        help="output path; defaults to me/videos/<example>.mp4. Relative paths resolve to the newton root.",
    )
    pre.add_argument("--num-frames", type=int, default=300)
    pre.add_argument("--fps", type=int, default=60)
    pre.add_argument("--width", type=int, default=1280)
    pre.add_argument("--height", type=int, default=720)
    pre.add_argument("--crf", type=int, default=18)
    pre.add_argument("--cam-keep", action="store_true", help="keep the example's own camera instead of side-framing.")
    pre.add_argument("--cam-azimuth", type=float, default=90.0, help="camera azimuth around up axis [deg]; 90 = side.")
    pre.add_argument("--cam-elevation", type=float, default=6.0, help="camera elevation above horizon [deg].")
    pre.add_argument("--cam-zoom", type=float, default=1.0, help="distance multiplier; <1 moves closer.")
    known, rest = pre.parse_known_args(argv)

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")

    out_name = known.out if known.out is not None else f"me/videos/{known.example}.mp4"
    out_path = Path(out_name)
    if not out_path.is_absolute():
        out_path = NEWTON_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    example_class = _resolve_example(known.example)

    parser = example_class.create_parser()
    args = parser.parse_args(rest)
    args.viewer = "gl"
    args.headless = True

    viewer = ViewerGL(width=known.width, height=known.height, headless=True)
    example = example_class(viewer, args)

    if not known.cam_keep:
        _frame_camera(viewer, example, known.cam_azimuth, known.cam_elevation, known.cam_zoom)

    first = viewer.get_frame(render_ui=False)
    height, width = int(first.shape[0]), int(first.shape[1])

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(known.fps),
            "-i",
            "-",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(known.crf),
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
    )

    buffer = None
    try:
        for frame_idx in range(known.num_frames):
            example.step()
            example.render()
            buffer = viewer.get_frame(target_image=buffer)
            ffmpeg.stdin.write(buffer.numpy().tobytes())
            if not args.quiet and frame_idx % 30 == 0:
                print(f"[capture] frame {frame_idx + 1}/{known.num_frames}", flush=True)
    finally:
        ffmpeg.stdin.close()
        ffmpeg.wait()
        viewer.close()

    seconds = known.num_frames / known.fps
    print(f"[capture] wrote {out_path} ({known.num_frames} frames, {seconds:.1f}s at {known.fps} fps)")


if __name__ == "__main__":
    main(sys.argv[1:])
