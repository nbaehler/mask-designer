"""
This is the script that is used for evaluating phases for physical or simulation forward model

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.

-----

$ python eval.py --channel=[0 or 1 or 2 or 3] --root_path=[some path]

"""

import imageio
import os
import skimage.io
import scipy.io as sio
from slm_controller.hardware import slm_devices, SLMParam

# import sys
import torch
import numpy as np
from slm_designer.hardware import cam_devices, CamParam

# import configargparse

from slm_designer.neural_holography.propagation_ASM import propagation_ASM
from slm_designer.neural_holography.augmented_image_loader import ImageLoader
import slm_designer.neural_holography.utils as utils
from slm_designer.neural_holography.modules import PhysicalProp
from slm_designer.neural_holography.propagation_model import ModelPropagate

from slm_designer.experimental_setup import (
    PhysicalParams,
    physical_params,
    slm_device,
    cam_device,
)

from slm_controller.hardware import (
    SLMParam,
    slm_devices,
)

# # Command line argument processing
# p = configargparse.ArgumentParser()
# p.add(
#     "-c",
#     "--config_filepath",
#     required=False,
#     is_config_file=True,
#     help="Path to config file.",
# )

# p.add_argument("--channel", type=int, default=1, help="red:0, green:1, blue:2, rgb:3")
# p.add_argument(
#     "--prop_model",
#     type=str,
#     default="ASM",
#     help="Type of propagation model for reconstruction: ASM / MODEL / CAMERA",
# )
# p.add_argument(
#     "--root_path",
#     type=str,
#     default="./phases",
#     help="Directory where test phases are being stored.",
# )
# p.add_argument(
#     "--prop_model_dir",
#     type=str,
#     default="./calibrated_models/",
#     help="Directory for the CITL-calibrated wave propagation models",
# )
# p.add_argument(
#     "--calibration_path",
#     type=str,
#     default=f"./calibration",
#     help="Directory where calibration phases are being stored.",
# )
def eval(channel, prop_model, root_path, prop_model_dir, calibration_path):
    slm_settle_time = physical_params[PhysicalParams.SLM_SETTLE_TIME]
    prop_dist = physical_params[PhysicalParams.PROPAGATION_DISTANCE]
    wavelength = physical_params[PhysicalParams.WAVELENGTH]

    # Parse
    # opt = p.parse_args()
    # channel = channel
    chs = range(channel) if channel == 3 else [channel]  # retrieve all channels if channel is 3
    chan_strs = ("red", "green", "blue", "rgb")

    run_id = f'{root_path.split("/")[-1]}_{prop_model}'  # {algorithm}_{prop_model}

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
        # precomputed_H = propagator(
        #     torch.empty(1, 1, *slm_res, 2),
        #     feature_size,
        #     wavelength,
        #     prop_dist,
        #     return_H=True,
        # ).to(device)

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
        blur = utils.make_kernel_gaussian(0.85, 3)
        propagator = ModelPropagate(
            distance=prop_dist, feature_size=feature_size, wavelength=wavelength, blur=blur,
        ).to(device)

        propagator.load_state_dict(
            torch.load(os.path.join(prop_model_dir, f"{chan_strs[c]}.pth"), map_location=device,)
        )
        propagator.eval()

    print(f"  - reconstruction with {prop_model}... ")

    # Data path
    data_path = "./data"
    recon_path = "./recon"

    # Augmented image loader (if you want to shuffle, augment dataset, put options accordingly.)
    image_loader = ImageLoader(
        data_path,
        channel=channel if channel < 3 else None,
        image_res=image_res,
        homography_res=roi_res,
        crop_to_homography=True,
        shuffle=False,
        vertical_flips=False,
        horizontal_flips=False,
    )

    # Placeholders for metrics
    psnrs = {"amp": [], "lin": [], "srgb": []}
    ssims = {"amp": [], "lin": [], "srgb": []}
    idxs = []

    # Loop over the dataset
    for target in image_loader:
        # get target image
        target_amp, target_res, target_filename = target
        target_path, target_filename = os.path.split(target_filename[0])
        target_idx = target_filename.split("_")[-1]
        target_amp = target_amp.to(device)

        print(f"    - running for img_{target_idx}...")

        # crop to ROI
        target_amp = utils.crop_image(target_amp, target_shape=roi_res, stacked_complex=False).to(
            device
        )

        recon_amp = []

        # for each channel, propagate wave from the SLM plane to the image plane and get the reconstructed image.
        for c in chs:
            # load and invert phase (our SLM setup)
            phase_filename = os.path.join(root_path, chan_strs[c], f"{target_idx}.png")
            slm_phase = skimage.io.imread(phase_filename) / 255.0
            slm_phase = (
                torch.tensor((1 - slm_phase) * 2 * np.pi - np.pi, dtype=dtype)
                .reshape(1, 1, *slm_res)
                .to(device)
            )

            # propagate field
            real, imag = utils.polar_to_rect(torch.ones_like(slm_phase), slm_phase)
            slm_field = torch.complex(real, imag)

            # if prop_model.upper() == "MODEL":
            #     propagator = propagators[
            #         c
            #     ]  # Select CITL-calibrated models for each channel
            recon_field = utils.propagate_field(
                slm_field, propagator, prop_dist, wavelength, feature_size, prop_model, dtype,
            )

            # cartesian to polar coordinate
            recon_amp_c = recon_field.abs()

            # crop to ROI
            recon_amp_c = utils.crop_image(recon_amp_c, target_shape=roi_res, stacked_complex=False)

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
        psnr_val, ssim_val = utils.get_psnr_ssim(recon_amp, target_amp, multichannel=(channel == 3))

        idxs.append(target_idx)

        for domain in ["amp", "lin", "srgb"]:
            psnrs[domain].append(psnr_val[domain])
            ssims[domain].append(ssim_val[domain])
            print(f"PSNR({domain}): {psnr_val[domain]},  SSIM({domain}): {ssim_val[domain]:.4f}, ")

        # save reconstructed image in srgb domain
        recon_srgb = utils.srgb_lin2gamma(np.clip(recon_amp ** 2, 0.0, 1.0))
        utils.cond_mkdir(recon_path)
        imageio.imwrite(
            os.path.join(recon_path, f"{target_idx}_{run_id}_{chan_strs[channel]}.png"),
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
