"""
Simulated propagation of phase masks generated using the SGD algorithm.
"""

from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)

import click
import numpy as np
from mask_designer.simulated_prop import simulated_prop
from mask_designer.utils import (
    build_field,
    pad_tensor_to_shape,
    show_fields,
)
from mask_designer.propagation import holoeye_fraunhofer, neural_holography_asm
from mask_designer.transform_fields import transform_from_neural_holography_setting
import torch

from slm_controller.hardware import (
    SLMDevices,
    SLMParam,
    slm_devices,
)
from mask_designer.experimental_setup import (
    Params,
    params,
    # slm_device,
)
from mask_designer.wrapper import SGD, ImageLoader
from mask_designer.utils import random_init_phase_mask


@click.command()
@click.option("--iterations", type=int, default=500, help="Number of iterations to run.")
def main(iterations):
    # Set parameters
    slm_device = SLMDevices.DUMMY.value
    # slm_device = SLMDevices.HOLOEYE_LC_2012.value

    prop_dist = params[Params.PROPAGATION_DISTANCE]
    wavelength = params[Params.WAVELENGTH]
    pixel_pitch = slm_devices[slm_device][SLMParam.PIXEL_PITCH]
    roi = params[Params.ROI]
    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]

    # Use GPU if detected in system
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize image loader
    image_loader = ImageLoader(
        abspath(join(CODE_DIR, "images/target_amplitude")),
        image_res=slm_shape,
        homography_res=roi,
        shuffle=False,
        vertical_flips=False,
        horizontal_flips=False,
    )

    # Load the the first image in the folder
    target_amp, _, _ = image_loader.load_image(0)

    # Make it grayscale
    target_amp = torch.mean(target_amp, axis=0)

    # Transform the image to be compliant with the neural holography data structure
    target_amp = target_amp[None, None, :, :]
    target_amp = target_amp.to(device)

    # Setup a random initial slm phase mask with values in [-0.5, 0.5]
    init_phase = random_init_phase_mask(slm_shape, device)

    # Run Stochastic Gradient Descent based method
    sgd = SGD(prop_dist, wavelength, pixel_pitch, iterations, roi, device=device)
    angles = sgd(target_amp, init_phase).cpu().detach()

    # Extend the computed angles, aka the phase values, to be a field which is a complex tensor again
    neural_holography_field = build_field(angles)

    # Transform the results to the hardware setting using a lens
    holoeye_field = transform_from_neural_holography_setting(
        neural_holography_field, prop_dist, wavelength, slm_shape, pixel_pitch
    )

    slm_device = SLMDevices.HOLOEYE_LC_2012.value
    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]

    angles = holoeye_field.angle()
    angles = pad_tensor_to_shape(angles, slm_shape, value=-np.pi + 0.00001)
    holoeye_field = build_field(angles)

    # Simulate the propagation in the lens setting and show the results
    unpacked_field = holoeye_field[0, 0, :, :]
    propped_field = simulated_prop(holoeye_field, holoeye_fraunhofer)
    show_fields(unpacked_field, propped_field, "Neural Holography SGD with lens")

    angles = neural_holography_field.angle()
    angles = pad_tensor_to_shape(angles, slm_shape, value=-np.pi + 0.00001)
    neural_holography_field = build_field(angles)

    # Simulate the propagation in the lensless setting and show the results
    unpacked_field = neural_holography_field[0, 0, :, :]
    propped_field = simulated_prop(
        neural_holography_field, neural_holography_asm, prop_dist, wavelength, pixel_pitch,
    )
    show_fields(unpacked_field, propped_field, "Neural Holography SGD without lens")


if __name__ == "__main__":
    main()