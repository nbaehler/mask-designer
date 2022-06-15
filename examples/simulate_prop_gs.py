"""
Simulated propagation of slm patterns generated using the GS algorithm.
"""

from slm_designer.utils import extend_to_complex, show_plot
from slm_designer.simulate_prop import lens_prop, lensless_prop
from slm_designer.transform_fields import lensless_to_lens
import torch

from slm_controller.hardware import (
    SLMDevices,
    SLMParam,
    slm_devices,
)
from slm_designer.hardware import (
    physical_params,
    PhysicalParams,
)
from slm_designer.neural_holography.modules import GS
from slm_designer.neural_holography.augmented_image_loader import ImageLoader

slm_device = SLMDevices.HOLOEYE_LC_2012.value


def simulate_prop_gs():
    # Set parameters
    distance = physical_params[PhysicalParams.PROPAGATION_DISTANCE]
    wavelength = physical_params[PhysicalParams.WAVELENGTH]
    feature_size = slm_devices[slm_device][SLMParam.CELL_DIM]
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

    # Run Gerchberg-Saxton
    gs = GS(distance, wavelength, feature_size, iterations, device=device)
    angles = gs(target_amp, init_phase).cpu().detach()

    # Extend the computed angles, aka the phase values, to a complex tensor again
    neural_holography_slm_field = extend_to_complex(angles)

    # Transform the results to the hardware setting using a lens
    temp = lensless_to_lens(neural_holography_slm_field)

    # Simulate the propagation in the lens setting and show the results
    slm_field = temp[0, 0, :, :]
    propped_slm_field = lens_prop(temp)[0, 0, :, :]
    show_plot(slm_field, propped_slm_field, "Neural Holography GS with lens")

    # Simulate the propagation in the lensless setting and show the results
    slm_field = neural_holography_slm_field[0, 0, :, :]
    propped_slm_field = lensless_prop(neural_holography_slm_field)[0, 0, :, :]
    show_plot(slm_field, propped_slm_field, "Neural Holography GS without lens")


if __name__ == "__main__":
    simulate_prop_gs()
