"""
Simulated propagation of slm patterns generated using the SGD algorithm.
"""

from slm_designer.utils import extend_to_complex, show_plot
from slm_designer.simulate_prop import lens_prop, lensless_prop
from slm_designer.transform_fields import lensless_to_lens
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
from slm_designer.wrapper import SGD, ImageLoader


def simulate_prop_sgd():
    # Set parameters
    distance = physical_params[PhysicalParams.PROPAGATION_DISTANCE]
    wavelength = physical_params[PhysicalParams.WAVELENGTH]
    pixel_pitch = slm_devices[slm_device][SLMParam.PIXEL_PITCH]
    iterations = 500

    slm_res = slm_devices[slm_device][SLMParam.SLM_SHAPE]
    image_res = slm_res
    roi_res = (round(slm_res[0] * 0.8), round(slm_res[1] * 0.8))

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

    # Load the the first image in the folder
    target_amp, _, _ = image_loader.load_image(0)
    target_amp = torch.mean(target_amp, axis=0)

    # Transform the image to be compliant with the neural holography data structure
    target_amp = target_amp[None, None, :, :]
    target_amp = target_amp.to(device)

    # Setup a random initial slm phase map with values in [-0.5, 0.5]
    init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *slm_res)).to(device)

    # Run Stochastic Gradient Descent based method
    sgd = SGD(distance, wavelength, pixel_pitch, iterations, roi_res, device=device)
    angles = sgd(target_amp, init_phase).cpu().detach()

    # Extend the computed angles, aka the phase values, to a complex tensor again
    neural_holography_slm_field = extend_to_complex(angles)

    # Transform the results to the hardware setting using a lens
    temp = lensless_to_lens(neural_holography_slm_field, distance, wavelength, slm_res, pixel_pitch)

    # Simulate the propagation in the lens setting and show the results
    slm_field = temp[0, 0, :, :]
    propped_slm_field = lens_prop(temp)[0, 0, :, :]
    show_plot(slm_field, propped_slm_field, "Neural Holography SGD with lens")

    # Simulate the propagation in the lensless setting and show the results
    slm_field = neural_holography_slm_field[0, 0, :, :]
    propped_slm_field = lensless_prop(
        neural_holography_slm_field,
        physical_params[PhysicalParams.PROPAGATION_DISTANCE],
        physical_params[PhysicalParams.WAVELENGTH],
        slm_devices[slm_device][SLMParam.PIXEL_PITCH],
    )[0, 0, :, :]
    show_plot(slm_field, propped_slm_field, "Neural Holography SGD without lens")


if __name__ == "__main__":
    simulate_prop_sgd()
