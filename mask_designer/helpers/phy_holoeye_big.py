"""
Physical propagation of the phase mask generated using the holoeye software.
"""

import sys
from os.path import abspath, dirname, join

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)

from mask_designer.utils import load_phase_mask
from slm_controller import slm
from slm_controller.hardware import SLMDevices


def main():  # TODO buggy, remove???
    # Load the phase mask generated using the holoeye software
    holoeye_phase_mask = load_phase_mask("images/test/holoeye_logo_big.png")

    # Initialize slm
    s = slm.create(SLMDevices.HOLOEYE_LC_2012.value)

    # display
    s.imshow(holoeye_phase_mask)


if __name__ == "__main__":
    main()
