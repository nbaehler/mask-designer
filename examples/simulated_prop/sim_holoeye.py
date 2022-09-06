"""
Simulated propagation of the phase mask generated using the holoeye software.
"""

import sys
from os.path import abspath, dirname, join

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)

from mask_designer.experimental_setup import Params, params, slm_device
from mask_designer.simulate_prop import (
    holoeye_fraunhofer,
    neural_holography_asm,
    plot_fields,
    simulate_prop,
)
from mask_designer.transform_fields import transform_to_neural_holography_setting
from mask_designer.utils import load_field
from slm_controller.hardware import SLMParam, slm_devices


def main():
    # Define parameters
    prop_dist = params[Params.PROPAGATION_DISTANCE]
    wavelength = params[Params.WAVELENGTH]
    pixel_pitch = slm_devices[slm_device][SLMParam.PIXEL_PITCH]
    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]

    # Load the field computed with holoeye software
    holoeye_field = load_field()

    # Make it compliant with the data structure used in the project
    unpacked_field = holoeye_field[0, 0, :, :]

    # Simulate the propagation in the lens setting and show the results
    propped_field = simulate_prop(holoeye_field, holoeye_fraunhofer)
    plot_fields(unpacked_field, propped_field, "Holoeye with lens")

    # Transform the initial field to the lensless setting
    neural_holography_field = transform_to_neural_holography_setting(
        holoeye_field, prop_dist, wavelength, slm_shape, pixel_pitch
    )
    unpacked_field = neural_holography_field[0, 0, :, :]

    # Simulate the propagation in the lensless setting and show the results
    propped_field = simulate_prop(
        neural_holography_field, neural_holography_asm, prop_dist, wavelength, pixel_pitch,
    )
    plot_fields(unpacked_field, propped_field, "Holoeye without lens")


if __name__ == "__main__":
    main()
