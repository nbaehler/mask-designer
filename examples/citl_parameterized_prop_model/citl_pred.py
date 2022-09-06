"""
Script that computes a phase mask using the CITL model.

This code is heavily inspired by mask_designer/neural_holography/eval.py. So
credit where credit is due.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.

Copyright (c) 2020, Stanford University

All rights reserved.

Refer to the LICENSE file for more information.
"""
import sys
from os.path import abspath, dirname, join

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)

import os
from pathlib import Path

import click
import imageio
import numpy as np
import skimage.io
import torch
from mask_designer import camera
from mask_designer.experimental_setup import (
    Params,
    amp_mask,
    cam_device,
    params,
    slm_device,
)
from mask_designer.utils import pad_tensor_to_shape
from mask_designer.wrapper import (
    PropModel,
    PropPhysical,
    cond_mkdir,
    crop_image,
    get_image_filenames,
    make_kernel_gaussian,
    polar_to_rect,
    propagate_field,
    prop_asm,
    srgb_lin2gamma,
)
from slm_controller import slm
from slm_controller.hardware import SLMParam, slm_devices


@click.command()
@click.option("--channel", type=int, default=1, help="red:0, green:1, blue:2, rgb:3")
@click.option(
    "--prop_model",
    type=str,
    default="ASM",
    help="Type of propagation model for reconstruction: ASM / MODEL / PHYSICAL",
)
@click.option(
    "--pred_phases_path",
    type=str,
    default="./citl/data/pred_phases",
    help="Directory where test phases are being stored.",
)
@click.option(
    "--prop_model_dir",
    type=str,
    default="./citl/calibrated_models",
    help="Directory for the CITL-calibrated wave propagation models",
)
@click.option(
    "--calibration_path",
    type=str,
    default="./citl/calibration",
    help="Directory where calibration phases are being stored.",
)
def main(
    channel, prop_model, pred_phases_path, prop_model_dir, calibration_path,
):
    slm_show_time = params[Params.SLM_SHOW_TIME]  # TODO arg or value from experimental setup
    slm_settle_time = params[Params.SLM_SETTLE_TIME]
    prop_dist = params[Params.PROPAGATION_DISTANCE]
    wavelength = params[Params.WAVELENGTH]
    roi = params[Params.ROI]

    chs = range(channel) if channel == 3 else [channel]  # retrieve all channels if channel is 3
    chan_strs = ("red", "green", "blue", "rgb")

    run_id = f'{pred_phases_path.split("/")[-1]}_{prop_model}'  # {algorithm}_{prop_model}

    # Hyperparameters setting
    prop_dists = (prop_dist, prop_dist, prop_dist)
    wavelengths = (wavelength, wavelength, wavelength)  # wavelength of each color
    feature_size = slm_devices[slm_device][SLMParam.PIXEL_PITCH]  # SLM pitch

    # Resolutions
    # slm_res = (1080, 1920)  # resolution of SLM
    # if "HOLONET" in run_id.upper():
    #     slm_res = (1072, 1920)
    # elif "UNET" in run_id.upper():
    #     slm_res = (1024, 2048)

    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]  # resolution of SLM

    dtype = torch.float32  # default datatype (results may differ if using, e.g., float64)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if prop_model == "ASM":
        propagator = prop_asm

    elif prop_model.upper() == "PHYSICAL":
        s = slm.create(slm_device)
        s.set_show_time(slm_show_time)

        cam = camera.create(cam_device)

        propagator = PropPhysical(
            s,
            slm_settle_time,
            cam,
            roi,
            channel,
            # # range_row=(220, 1000),
            # # range_col=(300, 1630),
            pattern_path=calibration_path,  # path of 12 x 21 calibration pattern, see Supplement.
            show_preview=True,
        )
    elif prop_model.upper() == "MODEL":
        blur = make_kernel_gaussian(0.85, 3)
        propagators = {}
        for c in chs:
            propagator = PropModel(
                distance=prop_dists[c],
                feature_size=feature_size,
                wavelength=wavelengths[c],
                blur=blur,
            ).to(device)

            propagator.load_state_dict(
                torch.load(
                    os.path.join(prop_model_dir, f"{chan_strs[c]}.pth"), map_location=device,
                )
            )
            propagator.eval()
            propagators[c] = propagator

    print(f"  - reconstruction with {prop_model}... ")

    # Data path
    pred_path = "./citl/predictions"

    images = get_image_filenames(pred_phases_path)

    slm_amp = amp_mask.to(device)

    # Loop over the dataset
    for pred_idx, phase_path in enumerate(images):
        amp = []

        # for each channel, propagate wave from the SLM plane to the image plane and get the reconstructed image.
        for c in chs:
            # load and invert phase (our SLM setup)
            phase_mask = skimage.io.imread(phase_path) / 255.0

            phase_mask = np.mean(phase_mask, axis=2)

            # phase_mask = (  #TODO inversion not needed in our setting?
            #     torch.tensor((1 - phase_mask) * 2 * np.pi - np.pi, dtype=dtype)
            #     .reshape(1, 1, *slm_shape)
            #     .to(device)
            # )

            phase_mask = (
                torch.tensor(phase_mask * 2 * np.pi - np.pi, dtype=dtype)
                .reshape(1, 1, *slm_shape)
                .to(device)
            )

            # propagate field
            real, imag = polar_to_rect(slm_amp, phase_mask)
            field = torch.complex(real, imag)

            if prop_model.upper() == "MODEL":
                propagator = propagators[c]  # Select CITL-calibrated models for each channel
            amp_map = propagate_field(
                field, propagator, prop_dists[c], wavelengths[c], feature_size, prop_model, dtype,
            )

            # cartesian to polar coordinate
            amp_c = amp_map.abs()

            # crop to ROI
            amp_c = crop_image(amp_c, target_shape=roi, stacked_complex=False)

            # append to list
            amp.append(amp_c)

        # list to tensor, scaling
        amp = torch.cat(amp, dim=1)

        amp = pad_tensor_to_shape(amp, slm_shape)  # TODO need to pad here?

        # tensor to numpy
        amp = amp.squeeze().cpu().detach().numpy()

        if channel == 3:
            amp = amp.transpose(1, 2, 0)

        # save reconstructed image in srgb domain
        srgb = srgb_lin2gamma(np.clip(amp ** 2, 0.0, 1.0))

        cond_mkdir(pred_path)
        imageio.imwrite(
            os.path.join(
                pred_path, f"{pred_idx}_{Path(phase_path).stem}_{run_id}_{chan_strs[channel]}.png",
            ),
            (srgb * np.iinfo(np.uint8).max).round().astype(np.uint8),
        )


if __name__ == "__main__":
    main()
