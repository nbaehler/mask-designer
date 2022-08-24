"""
Physical propagation of the phase mask generated using the holoeye software.
"""

from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)

from slm_controller.hardware import SLMDevices
from slm_controller import slm
from mask_designer.utils import load_phase_mask


def main():  # TODO buggy
    # Load the phase mask generated using the holoeye software
    holoeye_phase_mask = load_phase_mask("images/test/holoeye_logo_big.png")

    # Initialize slm
    s = slm.create(SLMDevices.HOLOEYE_LC_2012.value)

    # display
    s.imshow(holoeye_phase_mask)


if __name__ == "__main__":
    main()
