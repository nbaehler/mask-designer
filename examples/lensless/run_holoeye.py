"""
Propagation of the phase mask generated using the holoeye software.
"""

import sys
from os.path import abspath, dirname, join

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)

from mask_designer.experimental_setup import Params, params, slm_device
from mask_designer.simulate_prop import (
    neural_holography_asm,
    plot_fields,
    simulate_prop,
)
from mask_designer.transform_fields import holoeye_lens_to_lensless
from mask_designer.utils import load_field, quantize_phase_mask
from slm_controller import slm
from slm_controller.hardware import SLMDevices, SLMParam, slm_devices


def main():  # TODO does not work yet
    # Set parameters
    prop_distance = params[Params.PROPAGATION_DISTANCE]
    wavelength = params[Params.WAVELENGTH]
    pixel_pitch = slm_devices[slm_device][SLMParam.PIXEL_PITCH]
    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]

    # Load the field computed with holoeye software
    field = load_field()

    # Transform the initial field to the lensless setting
    field = holoeye_lens_to_lensless(field, prop_distance, wavelength, slm_shape, pixel_pitch)

    unpacked_field = field[0, 0, :, :]

    # Simulate the propagation in the lensless setting and show the results
    propped_field = simulate_prop(
        field, neural_holography_asm, prop_distance, wavelength, pixel_pitch,
    )
    plot_fields(unpacked_field, propped_field, "Holoeye without lens")

    # Quantize the fields angles, aka phase values, to a bit values
    phase = quantize_phase_mask(field.angle())

    # Initialize slm
    s = slm.create(SLMDevices.HOLOEYE_LC_2012.value)

    # display
    s.imshow(phase)


if __name__ == "__main__":
    main()
