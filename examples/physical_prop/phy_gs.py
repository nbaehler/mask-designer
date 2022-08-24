"""
Physical propagation of phase masks generated using the GS algorithm.
"""

from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)

import torch
import click
from slm_controller import slm
from slm_controller.hardware import (
    SLMParam,
    slm_devices,
)

from mask_designer.utils import random_init_phase_mask

from mask_designer.experimental_setup import (
    Params,
    params,
    slm_device,
)
from mask_designer.wrapper import ImageLoader
from mask_designer.methods import run_gs


@click.command()
@click.option("--iterations", type=int, default=500, help="Number of iterations to run.")
def main(iterations):
    # Set parameters
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

    # Run Gerchberg-Saxton
    phase_out = run_gs(
        init_phase, target_amp, iterations, slm_shape, prop_dist, wavelength, pixel_pitch, device,
    )

    # Instantiate SLM object
    s = slm.create(slm_device)

    # Display
    s.imshow(phase_out)


if __name__ == "__main__":
    main()
