"""
This is the script that is used for evaluating phases for physical or simulation forward model

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.

Copyright (c) 2020, Stanford University

All rights reserved.

Refer to the LICENSE file for more information.

$ python eval.py --channel=[0 or 1 or 2 or 3] --root_path=[some path]

"""

import os

import imageio
import mask_designer.neural_holography.utils as utils
import numpy as np
import scipy.io as sio
import skimage.io
import torch
from mask_designer import camera
from mask_designer.experimental_setup import (
    Params,
    amp_mask,
    cam_device,
    default_params,
    slm_device,
)
from mask_designer.neural_holography.augmented_image_loader import ImageLoader
from mask_designer.neural_holography.modules import PropPhysical
from mask_designer.neural_holography.prop_asm import prop_asm
from mask_designer.neural_holography.prop_model import PropModel
from slm_controller import slm
from slm_controller.hardware import SLMParam, slm_devices


def eval_model(
    channel, prop_model, test_phases_path, test_target_amps_path, prop_model_dir, calibration_path,
):
    slm_show_time = default_params[Params.SLM_SHOW_TIME]
    slm_settle_time = default_params[Params.SLM_SETTLE_TIME]
    prop_distance = default_params[Params.PROPAGATION_DISTANCE]
    wavelength = default_params[Params.WAVELENGTH]
    roi = default_params[Params.ROI]

    # Parse
    # opt = p.parse_args()
    # channel = channel
    chs = range(channel) if channel == 3 else [channel]  # retrieve all channels if channel is 3
    chan_strs = ("red", "green", "blue", "rgb")

    run_id = f'{test_phases_path.split("/")[-1]}_{prop_model}'  # {algorithm}_{prop_model}

    # Hyperparameters setting
    prop_dists = (prop_distance, prop_distance, prop_distance)
    wavelengths = (wavelength, wavelength, wavelength)  # wavelength of each color
    feature_size = slm_devices[slm_device][SLMParam.PIXEL_PITCH]  # SLM pitch

    # Resolutions
    # slm_shape = (1080, 1920)  # resolution of SLM
    # if "HOLONET" in run_id.upper():
    #     slm_shape = (1072, 1920)
    # elif "UNET" in run_id.upper():
    #     slm_shape = (1024, 2048)

    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]  # resolution of SLM

    dtype = torch.float32  # default datatype (results may differ if using, e.g., float64)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # You can pre-compute kernels for fast-computation
    # precomputed_H = [None] * 3

    if prop_model == "ASM":
        propagator = prop_asm
        # for c in chs:
        #     precomputed_H[c] = propagator(
        #         torch.empty(1, 1, *slm_shape, 2),
        #         feature_size,
        #         wavelengths[c],
        #         prop_dists[c],
        #         return_H=True,
        #     ).to(device)

    elif prop_model.upper() == "PHYSICAL":
        s = slm.create(slm_device)
        s.set_show_time(slm_show_time)

        cam = camera.create(cam_device)

        propagator = PropPhysical(
            s,
            slm_settle_time,
            slm_show_time,
            cam,
            roi,
            prop_distance,
            wavelength,
            channel,
            # range_row=(220, 1000),
            # range_col=(300, 1630),
            pattern_path=calibration_path,  # path of 12 x 21 calibration pattern, see Supplement.
            show_preview=True,
        )
    elif prop_model.upper() == "MODEL":
        blur = utils.make_kernel_gaussian(0.85, 3)
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
    recon_path = "./citl/reconstructions"

    # Augmented image loader (if you want to shuffle, augment dataset, put options accordingly.)
    image_loader = ImageLoader(
        test_target_amps_path,
        channel=channel if channel < 3 else None,
        image_res=slm_shape,
        homography_res=roi,
        crop_to_homography=True,
        shuffle=False,
        vertical_flips=False,
        horizontal_flips=False,
    )

    # Placeholders for metrics
    psnrs = {"amp": [], "lin": [], "srgb": []}
    ssims = {"amp": [], "lin": [], "srgb": []}
    idxs = []

    slm_amp = amp_mask.to(device)

    # Loop over the dataset
    for target_idx, target in enumerate(image_loader):
        # get target image
        target_amp, _, target_filename = target
        _, target_filename = os.path.split(target_filename[0])
        target_amp = target_amp.to(device)

        print(f"    - running for {target_filename}...")

        # crop to ROI
        target_amp = utils.crop_image(target_amp, target_shape=roi, stacked_complex=False).to(
            device
        )

        recon_amp = []

        # for each channel, propagate wave from the SLM plane to the image plane and get the reconstructed image.
        for c in chs:
            # load and invert phase (our SLM setup)
            phase_filename = os.path.join(test_phases_path, chan_strs[c], f"{target_filename}.png")
            phase_mask = skimage.io.imread(phase_filename) / 255.0

            phase_mask = np.mean(phase_mask, axis=2)

            # Inversion not needed in our setting
            # phase_mask = (
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
            real, imag = utils.polar_to_rect(slm_amp, phase_mask)
            slm_field = torch.complex(real, imag)

            if prop_model.upper() == "MODEL":
                propagator = propagators[c]  # Select CITL-calibrated models for each channel
            recon_field = utils.propagate_field(
                slm_field,
                propagator,
                prop_dists[c],
                wavelengths[c],
                feature_size,
                prop_model,
                dtype,
            )

            # cartesian to polar coordinate
            recon_amp_c = recon_field.abs()

            # crop to ROI
            recon_amp_c = utils.crop_image(recon_amp_c, target_shape=roi, stacked_complex=False)

            # append to list
            recon_amp.append(recon_amp_c)

        # list to tensor, scaling
        recon_amp = torch.cat(recon_amp, dim=1)
        recon_amp *= torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True) / torch.sum(
            recon_amp * recon_amp, (-2, -1), keepdim=True
        )

        # tensor to numpy
        recon_amp = recon_amp.squeeze().cpu().detach().numpy()
        target_amp = target_amp.squeeze().cpu().detach().numpy()

        if channel == 3:
            recon_amp = recon_amp.transpose(1, 2, 0)
            target_amp = target_amp.transpose(1, 2, 0)

        # calculate metrics
        psnr_val, ssim_val = utils.get_psnr_ssim(recon_amp, target_amp, channel_axis=(channel == 3))

        idxs.append(target_idx)

        for domain in ["amp", "lin", "srgb"]:
            psnrs[domain].append(psnr_val[domain])
            ssims[domain].append(ssim_val[domain])
            print(f"PSNR({domain}): {psnr_val[domain]},  SSIM({domain}): {ssim_val[domain]:.4f}, ")

        # save reconstructed image in srgb domain
        recon_srgb = utils.srgb_lin2gamma(np.clip(recon_amp ** 2, 0.0, 1.0))
        utils.cond_mkdir(recon_path)
        imageio.imwrite(
            os.path.join(
                recon_path, f"{target_idx}_{target_filename}_{run_id}_{chan_strs[channel]}.png",
            ),
            (recon_srgb * np.iinfo(np.uint8).max).round().astype(np.uint8),
        )

    # save it as a .mat file
    data_dict = {"img_idx": idxs}
    for domain in ["amp", "lin", "srgb"]:
        data_dict[f"ssims_{domain}"] = ssims[domain]
        data_dict[f"psnrs_{domain}"] = psnrs[domain]

    sio.savemat(
        os.path.join(recon_path, f"metrics_{run_id}_{chan_strs[channel]}.mat"), data_dict,
    )
