"""
Script that computes a phase map using the CITL model. TODO: not working entirely

This code is heavily inspired by slm_designer/neural_holography/eval.py. So
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

import click
import imageio
import os
import skimage.io
import torch
import numpy as np
from pathlib import Path


from slm_controller.hardware import (
    SLMParam,
    slm_devices,
)

from slm_designer.hardware import (
    CamParam,
    cam_devices,
)

from slm_designer.experimental_setup import (
    PhysicalParams,
    physical_params,
    slm_device,
    cam_device,
)

from slm_designer.wrapper import (
    ModelPropagate,
    propagation_ASM,
    get_image_filenames,
    make_kernel_gaussian,
    crop_image,
    polar_to_rect,
    propagate_field,
    srgb_lin2gamma,
    cond_mkdir,
    PhysicalProp,
)


@click.command()
@click.option("--channel", type=int, default=1, help="red:0, green:1, blue:2, rgb:3")
@click.option(
    "--prop_model",
    type=str,
    default="ASM",
    help="Type of propagation model for reconstruction: ASM / MODEL / CAMERA",
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
    default="./citl/calibrated_models",  # TODO normally calibrated in manual step?
    help="Directory for the CITL-calibrated wave propagation models",
)
@click.option(
    "--calibration_path",
    type=str,
    default="./citl/calibration",
    help="Directory where calibration phases are being stored.",
)
def citl_predict(
    channel, prop_model, pred_phases_path, prop_model_dir, calibration_path,
):
    slm_settle_time = physical_params[PhysicalParams.SLM_SETTLE_TIME]
    prop_dist = physical_params[PhysicalParams.PROPAGATION_DISTANCE]
    wavelength = physical_params[PhysicalParams.WAVELENGTH]

    # Parse
    # opt = p.parse_args()
    # channel = channel
    chs = range(channel) if channel == 3 else [channel]  # retrieve all channels if channel is 3
    chan_strs = ("red", "green", "blue", "rgb")

    run_id = f'{pred_phases_path.split("/")[-1]}_{prop_model}'  # {algorithm}_{prop_model}

    # Hyperparameters setting
    prop_dists = (prop_dist, prop_dist, prop_dist)
    wavelengths = (wavelength, wavelength, wavelength)  # wavelength of each color
    feature_size = slm_devices[slm_device][
        SLMParam.PIXEL_PITCH
    ]  # SLM pitch #TODO remove this dependency

    # Resolutions
    # slm_res = (1080, 1920)  # resolution of SLM
    # if "HOLONET" in run_id.upper():
    #     slm_res = (1072, 1920)
    # elif "UNET" in run_id.upper():
    #     slm_res = (1024, 2048)

    slm_res = slm_devices[slm_device][SLMParam.SLM_SHAPE]  # resolution of SLM
    image_res = cam_devices[cam_device][CamParam.IMG_SHAPE]  # TODO slm.shape == image.shape?
    roi_res = (round(slm_res[0] * 0.8), round(slm_res[1] * 0.8))

    dtype = torch.float32  # default datatype (results may differ if using, e.g., float64)
    device = "cuda" if torch.cuda.is_available() else "cpu"  # TODO gpu is too small
    # device = "cpu"

    # You can pre-compute kernels for fast-computation
    # precomputed_H = [None] * 3

    if prop_model == "ASM":
        propagator = propagation_ASM
        # for c in chs:
        #     precomputed_H[c] = propagator(
        #         torch.empty(1, 1, *slm_res, 2),
        #         feature_size,
        #         wavelengths[c],
        #         prop_dists[c],
        #         return_H=True,
        #     ).to(device)

    elif prop_model.upper() == "CAMERA":
        propagator = PhysicalProp(
            channel,
            laser_arduino=True,
            roi_res=(roi_res[1], roi_res[0]),
            slm_settle_time=slm_settle_time,
            range_row=(220, 1000),
            range_col=(300, 1630),
            patterns_path=calibration_path,  # path of 21 x 12 calibration patterns, see Supplement.
            show_preview=True,
        )
    elif prop_model.upper() == "MODEL":
        blur = make_kernel_gaussian(0.85, 3)
        propagators = {}
        for c in chs:
            propagator = ModelPropagate(
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
    # test_target_amps_path = "./citl/data/test_target_amps"
    pred_path = "./citl/predictions"

    # Augmented image loader (if you want to shuffle, augment dataset, put options accordingly.)
    # image_loader = ImageLoader(
    #     pred_phases_path,
    #     channel=channel if channel < 3 else None,
    #     image_res=image_res,
    #     homography_res=roi_res,
    #     crop_to_homography=True,
    #     shuffle=False,
    #     vertical_flips=False,
    #     horizontal_flips=False,
    # )

    # # Placeholders for metrics
    # psnrs = {"amp": [], "lin": [], "srgb": []}
    # ssims = {"amp": [], "lin": [], "srgb": []}
    # idxs = []

    images = get_image_filenames(pred_phases_path)

    # Loop over the dataset
    for pred_idx, phase_path in enumerate(images):
        # get target image
        # target_amp, _, target_filename = pred_phase
        # _, target_filename = os.path.split(target_filename[0])
        # # target_idx = target_filename.split("_")[-1]
        # target_amp = target_amp.to(device)

        # print(f"    - running for {target_filename}...")

        # # crop to ROI
        # target_amp = crop_image(
        #     target_amp, target_shape=roi_res, stacked_complex=False
        # ).to(device)

        pred_amp = []

        # for each channel, propagate wave from the SLM plane to the image plane and get the reconstructed image.
        for c in chs:
            # load and invert phase (our SLM setup)
            slm_phase = skimage.io.imread(phase_path) / 255.0

            slm_phase = np.mean(slm_phase, axis=2)  # TODO added to make it grayscale

            slm_phase = (
                torch.tensor((1 - slm_phase) * 2 * np.pi - np.pi, dtype=dtype)
                .reshape(1, 1, *slm_res)
                .to(device)
            )

            # propagate field
            real, imag = polar_to_rect(torch.ones_like(slm_phase), slm_phase)
            slm_field = torch.complex(real, imag)

            if prop_model.upper() == "MODEL":
                propagator = propagators[c]  # Select CITL-calibrated models for each channel
            pred_field = propagate_field(
                slm_field,
                propagator,
                prop_dists[c],
                wavelengths[c],
                feature_size,
                prop_model,
                dtype,
            )

            # cartesian to polar coordinate
            pred_amp_c = pred_field.abs()

            # crop to ROI
            pred_amp_c = crop_image(pred_amp_c, target_shape=roi_res, stacked_complex=False)

            # append to list
            pred_amp.append(pred_amp_c)

        # list to tensor, scaling
        pred_amp = torch.cat(pred_amp, dim=1)
        # pred_amp *= torch.sum(
        #     pred_amp * target_amp, (-2, -1), keepdim=True
        # ) / torch.sum(pred_amp * pred_amp, (-2, -1), keepdim=True)

        # tensor to numpy
        pred_amp = pred_amp.squeeze().cpu().detach().numpy()
        # target_amp = target_amp.squeeze().cpu().detach().numpy()

        if channel == 3:
            pred_amp = pred_amp.transpose(1, 2, 0)
            # target_amp = target_amp.transpose(1, 2, 0)

        # # calculate metrics
        # psnr_val, ssim_val = get_psnr_ssim(
        #     pred_amp, target_amp, channel_axis=(channel == 3)
        # )

        # idxs.append(target_idx)

        # for domain in ["amp", "lin", "srgb"]:
        #     psnrs[domain].append(psnr_val[domain])
        #     ssims[domain].append(ssim_val[domain])
        #     print(
        #         f"PSNR({domain}): {psnr_val[domain]},  SSIM({domain}): {ssim_val[domain]:.4f}, "
        #     )

        # save reconstructed image in srgb domain
        pred_srgb = srgb_lin2gamma(np.clip(pred_amp ** 2, 0.0, 1.0))
        cond_mkdir(pred_path)
        imageio.imwrite(
            os.path.join(
                pred_path, f"{pred_idx}_{Path(phase_path).stem}_{run_id}_{chan_strs[channel]}.png",
            ),
            (pred_srgb * np.iinfo(np.uint8).max).round().astype(np.uint8),
        )

    # # save it as a .mat file
    # data_dict = {"img_idx": idxs}
    # for domain in ["amp", "lin", "srgb"]:
    #     data_dict[f"ssims_{domain}"] = ssims[domain]
    #     data_dict[f"psnrs_{domain}"] = psnrs[domain]

    # sio.savemat(  # TODO why in mat format?
    #     os.path.join(recon_path, f"metrics_{run_id}_{chan_strs[channel]}.mat"),
    #     data_dict,
    # )


if __name__ == "__main__":
    citl_predict()
