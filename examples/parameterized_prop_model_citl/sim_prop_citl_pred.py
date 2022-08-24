"""
Simulated propagation of the phase mask generated using the holoeye software.
"""

from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)

import torch

from mask_designer.experimental_setup import Params, params, slm_device
from mask_designer.simulated_prop import simulated_prop

from mask_designer.utils import (
    angularize_phase_mask,
    build_field,
    load_phase_mask,
    pad_image_to_shape,
    show_fields,
)
from mask_designer.propagation import (
    holoeye_fraunhofer,
    neural_holography_asm,
)
from mask_designer.transform_fields import transform_to_neural_holography_setting

from slm_controller.hardware import SLMParam, slm_devices


def main():  # TODO buggy
    # Define parameters
    prop_dist = params[Params.PROPAGATION_DISTANCE]
    wavelength = params[Params.WAVELENGTH]
    pixel_pitch = slm_devices[slm_device][SLMParam.PIXEL_PITCH]
    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]

    # Load slm phase mask computed with CITL
    holoeye_phase_mask = load_phase_mask(
        "citl/predictions/0_holoeye_logo_pred_phases_ASM_green.png"
    )

    # Pad roi to full slm shape
    holoeye_phase_mask = torch.from_numpy(pad_image_to_shape(holoeye_phase_mask, slm_shape))[
        None, None, :, :
    ]  # TODO padding really needed? Done in citl_pred as well, we'll see

    holoeye_field = build_field(angularize_phase_mask(holoeye_phase_mask))
    unpacked_field = holoeye_field[0, 0, :, :]

    # Simulate the propagation in the lens setting and show the results
    propped_field = simulated_prop(holoeye_field, holoeye_fraunhofer)
    show_fields(unpacked_field, propped_field, "CITL with lens")

    # Transform the initial field to the lensless setting
    neural_holography_field = transform_to_neural_holography_setting(
        holoeye_field, prop_dist, wavelength, slm_shape, pixel_pitch
    )
    unpacked_field = neural_holography_field[0, 0, :, :]

    # Simulate the propagation in the lensless setting and show the results
    propped_field = simulated_prop(
        neural_holography_field, neural_holography_asm, prop_dist, wavelength, pixel_pitch,
    )
    show_fields(unpacked_field, propped_field, "CITL without lens")


if __name__ == "__main__":
    main()
