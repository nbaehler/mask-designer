"""
Physical propagation of slm patterns generated using the DPAC algorithm.
"""

import torch
import click
from slm_controller import slm
from slm_controller.hardware import (
    SLMDevices,
    SLMParam,
    slm_devices,
)

from physical_params import (
    PhysicalParams,
    physical_params,
)

from slm_designer.wrapper import ImageLoader, run_dpac

slm_device = SLMDevices.HOLOEYE_LC_2012.value


# Set parameters
distance = physical_params[PhysicalParams.PROPAGATION_DISTANCE]
wavelength = physical_params[PhysicalParams.WAVELENGTH]
pixel_pitch = slm_devices[slm_device][SLMParam.PIXEL_PITCH]
iterations = 500

slm_res = slm_devices[slm_device][SLMParam.SLM_SHAPE]
image_res = slm_res

roi_res = (round(slm_res[0] * 0.8), round(slm_res[1] * 0.8))


@click.command()
@click.option("--show_time", type=float, default=5.0, help="Time to show the pattern on the SLM.")
def physical_prop_dpac(show_time):
    # Use GPU if detected in system
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize image loader
    image_loader = ImageLoader(
        "images/target_amplitude",
        image_res=image_res,
        homography_res=roi_res,
        shuffle=False,
        vertical_flips=False,
        horizontal_flips=False,
    )

    # Instantiate SLM object
    s = slm.create_slm(slm_device)
    s.set_show_time(show_time)

    # Load the the first image in the folder
    target_amp, _, _ = image_loader.load_image(0)

    # Make it grayscale
    target_amp = torch.mean(target_amp, axis=0)

    # Transform the image to be compliant with the neural holography data structure
    target_amp = target_amp[None, None, :, :]
    target_amp = target_amp.to(device)

    # Run Double Phase Amplitude Coding #TODO does not work
    phase_out = run_dpac(target_amp, slm_res, distance, wavelength, pixel_pitch, device)

    # Display
    s.imshow(phase_out)


if __name__ == "__main__":
    physical_prop_dpac()
