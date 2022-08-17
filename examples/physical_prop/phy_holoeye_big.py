"""
Physical propagation of the slm pattern generated using the holoeye software.
"""

from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)

from slm_controller.hardware import SLMDevices
from slm_controller import slm
from mask_designer.utils import load_phase_map, quantize_phase_pattern


def main():
    # Load the slm pattern generated using the holoeye software
    holoeye_phase_map = load_phase_map("images/holoeye_logo_big.png").angle()
    holoeye_phase_map = quantize_phase_pattern(holoeye_phase_map)

    # Initialize slm
    s = slm.create(SLMDevices.HOLOEYE_LC_2012.value)

    # display
    s.imshow(holoeye_phase_map)


if __name__ == "__main__":
    main()
