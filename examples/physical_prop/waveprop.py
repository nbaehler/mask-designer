"""
Physical propagation of slm patterns generated using the SGD algorithm and a
waveprop propagator.
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

from mask_designer.experimental_setup import (
    Params,
    params,
    slm_device,
)

from mask_designer.propagation import propagator_waveprop_angular_spectrum
from mask_designer.transform_phase_maps import transform_from_neural_holography_setting
from mask_designer.utils import extend_to_complex, quantize_phase_pattern
from mask_designer.wrapper import ImageLoader, SGD


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

    # Instantiate SLM object
    s = slm.create(slm_device)

    # Load the the first image in the folder
    target_amp, _, _ = image_loader.load_image(0)

    # Make it grayscale
    target_amp = torch.mean(target_amp, axis=0)

    # Transform the image to be compliant with the neural holography data structure
    target_amp = target_amp[None, None, :, :]
    target_amp = target_amp.to(device)

    # Setup a random initial slm phase map with values in [-0.5, 0.5]
    init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *slm_shape)).to(device)

    # Run Stochastic Gradient Descent based method
    sgd = SGD(
        prop_dist,
        wavelength,
        pixel_pitch,
        iterations,
        roi,
        propagator=propagator_waveprop_angular_spectrum,
        device=device,
    )
    angles = sgd(target_amp, init_phase).cpu().detach()

    # Extend the computed angles, aka the phase values, to a complex tensor again
    extended = extend_to_complex(angles)

    # Transform the results to the hardware setting using a lens
    final_phase_sgd = transform_from_neural_holography_setting(
        extended, prop_dist, wavelength, slm_shape, pixel_pitch
    ).angle()

    # Quantize the the angles, aka phase values, to a bit values
    phase_out = quantize_phase_pattern(final_phase_sgd)

    # Display
    s.imshow(phase_out)


if __name__ == "__main__":
    main()
