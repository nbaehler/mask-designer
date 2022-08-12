"""
Physical propagation of the slm pattern generated using the holoeye software.
"""

from slm_controller.hardware import SLMDevices
from slm_controller import slm
from mask_designer.utils import load_phase_map, quantize_phase_pattern


def physical_prop_holoeye():
    # Load the slm pattern generated using the holoeye software
    holoeye_phase_map = load_phase_map().angle()
    holoeye_phase_map = quantize_phase_pattern(holoeye_phase_map)

    # Initialize slm
    s = slm.create_slm(SLMDevices.HOLOEYE_LC_2012.value)

    # display
    s.imshow(holoeye_phase_map)


if __name__ == "__main__":
    physical_prop_holoeye()
