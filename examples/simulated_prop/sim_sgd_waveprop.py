"""
Simulated propagation of phase masks generated using the SGD algorithm and the
angular spectrum propagator implemented in waveprop.
"""

import sys
from os.path import abspath, dirname, join

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)

import click
import torch
from mask_designer.experimental_setup import Params, params, slm_device
from mask_designer.prop_waveprop_asm import prop_waveprop_asm
from mask_designer.simulate_prop import (
    holoeye_fraunhofer,
    neural_holography_asm,
    plot_fields,
    simulate_prop,
    waveprop_asm,
)
from mask_designer.transform_fields import transform_from_neural_holography_setting
from mask_designer.utils import extend_to_field, random_init_phase_mask
from mask_designer.wrapper import SGD, ImageLoader
from slm_controller.hardware import SLMParam, slm_devices


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
    target_amp = torch.mean(target_amp, axis=0)

    # Transform the image to be compliant with the neural holography data structure
    target_amp = target_amp[None, None, :, :]
    target_amp = target_amp.to(device)

    # Setup a random initial slm phase mask with values in [-0.5, 0.5]
    init_phase = random_init_phase_mask(slm_shape, device)

    # Run Stochastic Gradient Descent based method
    sgd = SGD(
        prop_dist,
        wavelength,
        pixel_pitch,
        iterations,
        roi,
        propagator=prop_waveprop_asm,  # TODO this propagator is not in the neural holography setting
        device=device,
    )
    angles = sgd(target_amp, init_phase).cpu().detach()

    # Extend the computed angles, aka the phase values, to be a field which is a complex tensor again
    field = extend_to_field(angles)

    # Transform the results to the hardware setting using a lens
    holoeye_field = transform_from_neural_holography_setting(
        field, prop_dist, wavelength, slm_shape, pixel_pitch
    )

    # Simulate the propagation in the lens setting and show the results
    unpacked_field = holoeye_field[0, 0, :, :]
    propped_field = simulate_prop(holoeye_field, holoeye_fraunhofer)
    plot_fields(
        unpacked_field, propped_field, "Neural Holography SGD with lens and Holoeye Fraunhofer",
    )

    # Simulate the propagation in the lensless setting and show the results
    unpacked_field = field[0, 0, :, :]
    propped_field = simulate_prop(field, neural_holography_asm, prop_dist, wavelength, pixel_pitch,)
    plot_fields(
        unpacked_field,
        propped_field,
        "Neural Holography SGD without lens and Neural Holography ASM",
    )

    # Simulate the propagation in the lens setting and show the results
    unpacked_field = holoeye_field[0, 0, :, :]
    propped_field = (
        simulate_prop(holoeye_field, waveprop_asm, prop_dist, wavelength, pixel_pitch, device,)
        .cpu()
        .detach()
    )
    plot_fields(
        unpacked_field, propped_field, "Neural Holography SGD with lens and waveprop ASM",
    )


if __name__ == "__main__":
    main()
