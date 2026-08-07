"""Re-measure the finest drop configuration on a second GPU to test reproducibility.

`shank_parity/drop_contact.py` records every entry on one device. This runs the same two
32 x 80 configurations wherever CUDA_VISIBLE_DEVICES points, so the overdamped and well-posed
coefficients can be compared across hardware from identical code and inputs. Results are keyed
by device and merged into OUT_JSON, so each device is recorded once:

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n> python <this>

CUDA_DEVICE_ORDER matters -- the default is FASTEST_FIRST, which does not match nvidia-smi.
"""

import json
import os

import warp as wp

import newton.viewer
from newton.examples.robot import example_robot_compliant_shank_drop as ex

OUT_JSON = "/tmp/claude-1000/newton-sparse-tests/me/data/shank_parity/drop_device_check.json"
SUBSTEPS = 32
ITERATIONS = 80
CONTACT_KD = [1.0e5, 0.35]
RELAXATION = 0.8
CALIBRATED_YOUNG_MODULUS = 9.0e10
MATCHED_DAMPING_RATIO = 0.02


class Args:
    centerline = None


results = {}
if os.path.exists(OUT_JSON):
    with open(OUT_JSON) as handle:
        results = json.load(handle)

for contact_kd in CONTACT_KD:
    ex.SUBSTEPS = SUBSTEPS
    ex.ITERATIONS = ITERATIONS
    ex.ARTICULATION_RELAXATION = RELAXATION
    ex.BLADE_YOUNG_MODULUS = CALIBRATED_YOUNG_MODULUS
    ex.BLADE_DAMPING_RATIO = MATCHED_DAMPING_RATIO
    ex.PRDM_CONTACT_KD = contact_kd
    ex.BLADE_CONTACT_KD = contact_kd
    frames = int(ex.DURATION * ex.FPS)
    example = ex.Example(newton.viewer.ViewerNull(num_frames=frames), Args)
    for _ in range(frames):
        example.step()
    key = f"{wp.get_device().name}|kd={contact_kd:g}|{SUBSTEPS}x{ITERATIONS}"
    results[key] = {"device": wp.get_device().name}
    for row in example.summary():
        results[key][row["name"]] = {
            "squash": row["compression"] * 1e3,
            "rebound_height": row["rebound_height"] * 1e3,
        }
    entry = results[key]
    print(
        f"{key:22s} {entry['device']:20s} "
        f"chain {entry['PRDM']['squash']:6.2f} / {entry['PRDM']['rebound_height']:6.2f}   "
        f"blade {entry['elastic']['squash']:6.2f} / {entry['elastic']['rebound_height']:6.2f}",
        flush=True,
    )

with open(OUT_JSON, "w") as handle:
    json.dump(results, handle, indent=2)
print(f"wrote {OUT_JSON}")
