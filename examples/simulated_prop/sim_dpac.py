"""
Simulated propagation of slm patterns generated using the DPAC algorithm.
"""
from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)


from mask_designer.simulated_prop import simulated_prop
from mask_designer.utils import extend_to_complex, show_plot
from mask_designer.propagation import holoeye_fraunhofer, neural_holography_asm
from mask_designer.transform_phase_maps import transform_from_neural_holography_setting
import torch

from slm_controller.hardware import (
    SLMParam,
    slm_devices,
)
from mask_designer.experimental_setup import (
    Params,
    params,
    slm_device,
)
from mask_designer.wrapper import DPAC, ImageLoader


def main():
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

    # Run Double Phase Amplitude Coding #TODO DPAC does not work
    dpac = DPAC(prop_dist, wavelength, pixel_pitch, device=device)
    angles = dpac(target_amp)
    angles = angles.cpu().detach()

    # Extend the computed angles, aka the phase values, to a complex tensor again
    neural_holography_phase_map = extend_to_complex(angles)

    # Transform the results to the hardware setting using a lens
    holoeye_phase_map = transform_from_neural_holography_setting(
        neural_holography_phase_map, prop_dist, wavelength, slm_shape, pixel_pitch
    )

    # Simulate the propagation in the lens setting and show the results
    unpacked_phase_map = holoeye_phase_map[0, 0, :, :]
    propped_phase_map = simulated_prop(holoeye_phase_map, holoeye_fraunhofer)
    show_plot(unpacked_phase_map, propped_phase_map, "Neural Holography DPAC with lens")

    # Simulate the propagation in the lensless setting and show the results
    unpacked_phase_map = neural_holography_phase_map[0, 0, :, :]
    propped_phase_map = simulated_prop(
        neural_holography_phase_map, neural_holography_asm, prop_dist, wavelength, pixel_pitch,
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Neural Holography DPAC without lens")


if __name__ == "__main__":
    main()
