"""
Propagation of the phase mask generated using the holoeye software.
"""

import sys
from os.path import abspath, dirname, join

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)

from mask_designer.simulate_prop import holoeye_fraunhofer, plot_fields, simulate_prop
from mask_designer.utils import load_field, quantize_phase_mask
from slm_controller import slm
from slm_controller.hardware import SLMDevices


def main():
    # Load the field computed with holoeye software
    field = load_field()

    # Make it compliant with the data structure used in the project
    unpacked_field = field[0, 0, :, :]

    # Simulate the propagation in the lens setting and show the results
    propped_field = simulate_prop(field, holoeye_fraunhofer)
    plot_fields(unpacked_field, propped_field, "Holoeye with lens")

    # Quantize the fields angles, aka phase values, to a bit values
    phase = quantize_phase_mask(field.angle())

    # Initialize slm
    s = slm.create(SLMDevices.HOLOEYE_LC_2012.value)

    # display
    s.imshow(phase)


if __name__ == "__main__":
    main()
