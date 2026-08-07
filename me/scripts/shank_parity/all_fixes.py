"""Run both examples with every fix from sections 1-4 applied at once.

The report's other figures each isolate one change. These two are the end state: the step
example and the drop example with the relaxation raised, the step refined, the blade
recalibrated, the damping matched at the fundamental, and the chain's foot damping sized
against its own contact.
"""

import newton.viewer
from newton.examples.robot import example_robot_compliant_shank_drop as drop
from newton.examples.robot import example_robot_compliant_shank_step as step

FIGURES = "/tmp/claude-1000/newton-sparse-tests/me/figures/shank_parity"
RELAXATION = 0.8
SUBSTEPS = 8
CALIBRATED_YOUNG_MODULUS = 9.0e10
MATCHED_DAMPING_RATIO = 0.02
BLADE_CONTACT_KD = 0.35
STEP_DURATION = 1.05


class Args:
    centerline = None


def configure(module):
    module.ARTICULATION_RELAXATION = RELAXATION
    module.SUBSTEPS = SUBSTEPS
    module.BLADE_YOUNG_MODULUS = CALIBRATED_YOUNG_MODULUS
    module.BLADE_DAMPING_RATIO = MATCHED_DAMPING_RATIO


def run(module, frames):
    example = module.Example(newton.viewer.ViewerNull(num_frames=frames), Args)
    for _ in range(frames):
        example.step()
    return example


print("===== step example, all fixes =====")
configure(step)
example = run(step, int(STEP_DURATION * step.FPS))
example.print_summary()
example.plot(f"{FIGURES}/step_all_fixes.png")

print("\n===== drop example, all fixes =====")
configure(drop)
drop.PRDM_CONTACT_KD = BLADE_CONTACT_KD
example = run(drop, int(drop.DURATION * drop.FPS))
example.print_summary()
example.plot(f"{FIGURES}/drop_all_fixes.png")
