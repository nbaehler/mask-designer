"""
Physical propagation of slm patterns generated using the SGD algorithm.
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
from mask_designer import camera
from mask_designer.simulated_prop import simulated_prop


from mask_designer.experimental_setup import (
    Params,
    params,
    slm_device,
    cam_device,
)
from mask_designer.propagation import neural_holography_asm

from mask_designer.transform_phase_maps import transform_from_neural_holography_setting
from mask_designer.utils import extend_to_complex, quantize_phase_pattern, show_plot
from mask_designer.wrapper import ImageLoader, SGD, PhysicalProp


@click.command()
@click.option("--iterations", type=int, default=10, help="Number of iterations to run.")
@click.option(
    "--slm_show_time",  # TODO what makes sense to keep as arguments here and what should be moved to the experimental setup? Maybe we could just store the default values there ...
    type=float,
    default=params[Params.SLM_SHOW_TIME],
    help="Time to show the pattern on the SLM.",
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

    # Setup a random initial slm phase map with values in [-0.5, 0.5]
    init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *slm_shape)).to(device)

    # Run Stochastic Gradient Descent based method
    sgd = SGD(prop_dist, wavelength, pixel_pitch, 500, roi, device=device)
    angles = sgd(target_amp, init_phase).cpu().detach()

    # Extend the computed angles, aka the phase values, to a complex tensor again
    extended = extend_to_complex(angles)

    # Transform the results to the hardware setting using a lens
    warm_start_phase = transform_from_neural_holography_setting(
        extended, prop_dist, wavelength, slm_shape, pixel_pitch
    )

    unpacked_phase_map = warm_start_phase[0, 0, :, :]
    propped_phase_map = simulated_prop(
        warm_start_phase, neural_holography_asm, prop_dist, wavelength, pixel_pitch,
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Neural Holography GS without lens")

    warm_start_phase = warm_start_phase.angle().to(device)

    # --------------------------------------------------------------------------
    # TODO fix

    from multiprocessing.managers import BaseManager

    BaseManager.register("HoloeyeSLM", slm.HoloeyeSLM)  # TODO shouldn't to be shared
    BaseManager.register("IDSCamera", camera.IDSCamera)
    # BaseManager.register("DummyCamera", camera.DummyCamera)

    manager = BaseManager()
    manager.start()

    s = manager.HoloeyeSLM()  # TODO shouldn't to be shared
    # s = slm.create(slm_device)
    # s = None

    s.set_show_time(slm_show_time)

    cam = manager.IDSCamera()
    # cam = manager.DummyCamera()
    # cam = camera.create(cam_device)
    # cam = None

    cam.set_exposure_time(1200)

    # --------------------------------------------------------------------------

    import os
    import glob

    files = glob.glob("citl/snapshots/*.png")
    for f in files:
        os.remove(f)

    camera_prop = PhysicalProp(
        s,
        slm_settle_time,
        cam,
        roi_res=roi,
        # channel,
        # laser_arduino=True,
        # range_row=(220, 1000),
        # range_col=(300, 1630),
        # patterns_path=calibration_path,  # path of 12 x 21 calibration patterns, see Supplement.
        show_preview=True,
    )

    # Run Stochastic Gradient Descent based method
    sgd = SGD(
        prop_dist,
        wavelength,
        pixel_pitch,
        iterations,
        roi,
        device=device,
        citl=True,
        camera_prop=camera_prop,
    )
    angles = sgd(target_amp, warm_start_phase).cpu().detach()

    # Extend the computed angles, aka the phase values, to a complex tensor again
    extended = extend_to_complex(angles)

    # Transform the results to the hardware setting using a lens
    final_phase_sgd = transform_from_neural_holography_setting(
        extended, prop_dist, wavelength, slm_shape, pixel_pitch
    )

    unpacked_phase_map = final_phase_sgd[0, 0, :, :]
    propped_phase_map = simulated_prop(
        final_phase_sgd, neural_holography_asm, prop_dist, wavelength, pixel_pitch,
    )
    show_plot(unpacked_phase_map, propped_phase_map, "Neural Holography GS without lens")

    # Quantize the the angles, aka phase values, to a bit values
    phase_out = quantize_phase_pattern(final_phase_sgd.angle())

    # Display
    s.imshow(phase_out)
    final_res = cam.acquire_single_image()

    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    ax.imshow(final_res, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
