"""
Simulated propagation of slm patterns generated using the GS algorithm.
"""

import click
from slm_designer.simulated_prop import simulated_prop
from slm_designer.utils import extend_to_complex, show_plot
from slm_designer.propagation import holoeye_fraunhofer, neural_holography_asm
from slm_designer.transform_phase_maps import lensless_to_lens
import torch

from slm_controller.hardware import (
    SLMParam,
    slm_devices,
)
from slm_designer.experimental_setup import (
    PhysicalParams,
    physical_params,
    slm_device,
)
from slm_designer.wrapper import GS, ImageLoader


@click.command()
@click.option("--iterations", type=int, default=500, help="Number of iterations to run.")
def simulated_prop_gs(iterations):
    # Set parameters
    prop_dist = physical_params[PhysicalParams.PROPAGATION_DISTANCE]
    wavelength = physical_params[PhysicalParams.WAVELENGTH]
    pixel_pitch = slm_devices[slm_device][SLMParam.PIXEL_PITCH]
    roi = physical_params[PhysicalParams.ROI]
    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]

    # Use GPU if detected in system
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize image loader
    image_loader = ImageLoader(
        "images/target_amplitude",
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

    # Setup a random initial slm phase map with values in [-0.5, 0.5]
    init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *slm_shape)).to(device)

    # Run Gerchberg-Saxton
    gs = GS(prop_dist, wavelength, pixel_pitch, iterations, device=device)
    angles = gs(target_amp, init_phase).cpu().detach()

    # Extend the computed angles, aka the phase values, to a complex tensor again
    neural_holography_phase_map = extend_to_complex(angles)

    # Transform the results to the hardware setting using a lens
    holoeye_phase_map = lensless_to_lens(
        neural_holography_phase_map, prop_dist, wavelength, slm_shape, pixel_pitch
    )

    # Simulate the propagation in the lens setting and show the results
    unpacked_phase_map = holoeye_phase_map[0, 0, :, :]
    propped_phase_map = simulated_prop(holoeye_phase_map, holoeye_fraunhofer)
    show_plot(unpacked_phase_map, propped_phase_map, "Neural Holography GS with lens")

    # Simulate the propagation in the lensless setting and show the results
    unpacked_phase_map = neural_holography_phase_map[0, 0, :, :]
    propped_phase_map = simulated_prop(
        neural_holography_phase_map, neural_holography_asm, prop_dist, wavelength, pixel_pitch,
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Neural Holography GS without lens")


if __name__ == "__main__":
    simulated_prop_gs()
