"""
Physical propagation of slm patterns generated using the DPAC algorithm.
"""

import torch
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

from mask_designer.wrapper import ImageLoader, run_dpac


def physical_prop_dpac():
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
        "images/target_amplitude",
        image_res=slm_shape,
        homography_res=roi,
        shuffle=False,
        vertical_flips=False,
        horizontal_flips=False,
    )

    # Instantiate SLM object
    s = slm.create_slm(slm_device)

    # Load the the first image in the folder
    target_amp, _, _ = image_loader.load_image(0)

    # Make it grayscale
    target_amp = torch.mean(target_amp, axis=0)

    # Transform the image to be compliant with the neural holography data structure
    target_amp = target_amp[None, None, :, :]
    target_amp = target_amp.to(device)

    # Run Double Phase Amplitude Coding #TODO DPAC does not work
    phase_out = run_dpac(target_amp, slm_shape, prop_dist, wavelength, pixel_pitch, device)

    # Display
    s.imshow(phase_out)


if __name__ == "__main__":
    physical_prop_dpac()
