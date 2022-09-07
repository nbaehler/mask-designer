"""
Physical propagation of phase masks generated using the SGD algorithm.
"""


import sys
from os.path import abspath, dirname, join

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)

import datetime
import glob
import os
from multiprocessing.managers import BaseManager

import click
import torch
from mask_designer import camera
from mask_designer.experimental_setup import Params, params, slm_device
from mask_designer.prop_waveprop_asm import prop_waveprop_asm, prop_waveprop_asm_lens
from mask_designer.simulate_prop import holoeye_fraunhofer, simulate_prop
from mask_designer.transform_fields import transform_from_neural_holography_setting
from mask_designer.utils import (
    extend_to_field,
    quantize_phase_mask,
    random_init_phase_mask,
    save_image,
)
from mask_designer.wrapper import SGD, ImageLoader, PropPhysical, prop_asm
from slm_controller import slm
from slm_controller.hardware import SLMParam, slm_devices


@click.command()
@click.option(
    "--iterations", type=int, default=50, help="Number of iterations to run."
)  # TODO have two iteration arguments
@click.option(
    "--slm_show_time",  # TODO what makes sense to keep as arguments here and what should
    # be moved to the experimental setup? Maybe we could just store the default values there ...
    type=float,
    default=params[Params.SLM_SHOW_TIME],
    help="Time to show the mask on the SLM.",
)
@click.option(
    "--slm_settle_time",
    type=float,
    default=params[Params.SLM_SETTLE_TIME],
    help="Time to let the SLM to settle before taking images of the amplitude at the target plane.",
)
def main(iterations, slm_show_time, slm_settle_time):
    # Set parameters
    prop_dist = params[Params.PROPAGATION_DISTANCE]
    wavelength = params[Params.WAVELENGTH]
    pixel_pitch = slm_devices[slm_device][SLMParam.PIXEL_PITCH]
    roi = params[Params.ROI]
    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]

    asm_propagator = prop_asm
    # asm_propagator = prop_waveprop_asm # TODO check if this is correct
    # asm_propagator = prop_waveprop_asm_lens

    # warm_start_iterations = 500 # TODO use those
    # citl_iterations = iterations

    warm_start_iterations = 50
    citl_iterations = 5

    # Use GPU if detected in system
    device = "cuda" if torch.cuda.is_available() else "cpu"

    files = glob.glob("citl/snapshots/*.png")
    for f in files:
        os.remove(f)

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
    sgd = SGD(
        prop_dist,
        wavelength,
        pixel_pitch,
        warm_start_iterations,
        roi,
        device=device,
        propagator=asm_propagator,
    )
    angles = sgd(target_amp, init_phase).cpu().detach()

    # Extend the computed angles, aka the phase values, to be a field which is a complex tensor
    # again
    warm_start_field = extend_to_field(angles)

    # Transform the results to the hardware setting using a lens
    final_phase_sgd = transform_from_neural_holography_setting(
        warm_start_field, prop_dist, wavelength, slm_shape, pixel_pitch
    )

    propped_field = simulate_prop(final_phase_sgd, holoeye_fraunhofer)

    import numpy as np
    from mask_designer.utils import normalize_mask

    name = str(datetime.datetime.now().time()).replace(":", "_").replace(".", "_")

    print(  # TODO remove
        "propped_field.abs()",
        torch.min(propped_field.abs()).item(),
        torch.max(propped_field.abs()).item(),
        torch.median(propped_field.abs()).item(),
        torch.mean(propped_field.abs()).item(),
        torch.quantile(propped_field.abs(), 0.99).item(),
    )

    # save_image( # TODO remove
    #     propped_field.abs().numpy(), f"citl/snapshots/sim_{name}_warm_start.png",
    # )

    save_image(
        (255 * normalize_mask(propped_field.abs())).astype(
            np.uint8
        ),  # TODO check this version using normalization and cap using quantile
        f"citl/snapshots/sim_{name}_warm_start.png",
    )

    # Quantize the fields angles, aka phase values, to a bit values
    phase_out = quantize_phase_mask(final_phase_sgd.angle())

    BaseManager.register("HoloeyeSLM", slm.HoloeyeSLM)  # TODO shouldn't to needed to be shared
    BaseManager.register("IDSCamera", camera.IDSCamera)

    manager = BaseManager()
    manager.start()

    s = manager.HoloeyeSLM()
    s.set_show_time(slm_show_time)

    cam = manager.IDSCamera()
    cam.set_exposure_time(900)

    # Display
    s.imshow(phase_out)

    final_res = cam.acquire_single_image()

    name = str(datetime.datetime.now().time()).replace(":", "_").replace(".", "_")
    save_image(
        final_res, f"citl/snapshots/phy_{name}_warm_start.png",
    )

    warm_start_phase = warm_start_field.angle().to(device)

    prop_physical = PropPhysical(
        s,
        slm_settle_time,
        cam,
        roi_res=roi,
        # channel,
        # range_row=(220, 1000),
        # range_col=(300, 1630),
        # pattern_path=calibration_path,  # path of 12 x 21 calibration pattern, see Supplement.
        show_preview=True,
    )

    # Run Stochastic Gradient Descent based method
    sgd = SGD(
        prop_dist,
        wavelength,
        pixel_pitch,
        citl_iterations,
        roi,
        device=device,
        prop_model="PHYSICAL",
        propagator=prop_physical,
    )
    angles = sgd(target_amp, warm_start_phase).cpu().detach()

    # Extend the computed angles, aka the phase values, to be a field which is a complex tensor
    # again
    extended = extend_to_field(angles)

    # Transform the results to the hardware setting using a lens
    final_phase_sgd = transform_from_neural_holography_setting(
        extended, prop_dist, wavelength, slm_shape, pixel_pitch
    )

    propped_field = simulate_prop(final_phase_sgd, holoeye_fraunhofer)

    name = str(datetime.datetime.now().time()).replace(":", "_").replace(".", "_")

    # save_image( # TODO remove
    #     propped_field.abs().numpy(), f"citl/snapshots/sim_{name}_final.png",
    # )

    save_image(
        (255 * normalize_mask(propped_field.abs())).astype(
            np.uint8
        ),  # TODO check this version using normalization and cap using quantile
        f"citl/snapshots/sim_{name}_final.png",
    )

    # Quantize the fields angles, aka phase values, to a bit values
    phase_out = quantize_phase_mask(final_phase_sgd.angle())

    # Display
    s.imshow(phase_out)

    final_res = cam.acquire_single_image()

    name = str(datetime.datetime.now().time()).replace(":", "_").replace(".", "_")
    save_image(final_res, f"citl/snapshots/phy_{name}_final.png")


if __name__ == "__main__":
    main()
