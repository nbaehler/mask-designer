"""
Neural holography:

This is the main executive script used for training our parameterized wave propagation model

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

@article{Peng:2020:NeuralHolography,
author = {Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein},
title = {{Neural Holography with Camera-in-the-loop Training}},
journal = {ACM Trans. Graph. (SIGGRAPH Asia)},
year = {2020},
}

Copyright (c) 2020, Stanford University

All rights reserved.

Refer to the LICENSE file for more information.
-----

$ python train_model.py --channel=1 --experiment=test

"""

import os
import cv2

# import sys

# import time
import torch
import numpy as np

# import configargparse
import skimage.util
import torch.nn as nn
import torch.optim as optim
from PIL import Image

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

import slm_designer.neural_holography.utils as utils
from slm_designer.neural_holography.modules import PhysicalProp
from slm_designer.neural_holography.propagation_model import ModelPropagate
from slm_designer.neural_holography.augmented_image_loader import ImageLoader
from slm_designer.neural_holography.utils_tensorboard import SummaryModelWriter

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
#     "--pretrained_path",
#     type=str,
#     default="",
#     help="Path of pretrained checkpoints as a starting point.",
# )
# p.add_argument(
#     "--model_path",
#     type=str,
#     default="./models",
#     help="Directory for saving out checkpoints",
# )
# p.add_argument(
#     "--phase_path",
#     type=str,
#     default="./precomputed_phases",
#     help="Directory for precalculated phases",
# )
# p.add_argument(
#     "--calibration_path",
#     type=str,
#     default=f"./calibration",
#     help="Directory where calibration phases are being stored.",
# )
# p.add_argument(
#     "--lr_model", type=float, default=3e-3, help="Learning rate for model parameters"
# )
# p.add_argument("--lr_phase", type=float, default=5e-3, help="Learning rate for phase")
# p.add_argument("--num_epochs", type=int, default=15, help="Number of epochs")
# p.add_argument("--batch_size", type=int, default=2, help="Size of minibatch")
# p.add_argument(
#     "--step_lr", type=utils.str2bool, default=True, help="Use of lr scheduler"
# )
# p.add_argument("--experiment", type=str, default="", help="Name of the experiment")


# # parse arguments
# opt = p.parse_args()


def train_model(
    channel,
    pretrained_path,
    model_path,
    phase_path,
    calibration_path,
    train_data_path,
    lr_model,
    lr_phase,
    num_epochs,
    batch_size,
    step_lr,
    experiment,
):
    slm_settle_time = physical_params[PhysicalParams.SLM_SETTLE_TIME]
    prop_dist = physical_params[PhysicalParams.PROPAGATION_DISTANCE]
    wavelength = physical_params[PhysicalParams.WAVELENGTH]

    # channel = channel  # Red:0 / Green:1 / Blue:2
    chan_str = ("red", "green", "blue")[channel]
    run_id = f"{chan_str}_{experiment}_lr{lr_model}_batchsize{batch_size}"  # {algorithm}_{prop_model} format

    print(f"   - training parameterized wave propagation model....")

    feature_size = slm_devices[slm_device][
        SLMParam.PIXEL_PITCH
    ]  # SLM pitch #TODO remove this dependency

    slm_res = slm_devices[slm_device][SLMParam.SLM_SHAPE]  # resolution of SLM
    image_res = cam_devices[cam_device][CamParam.IMG_SHAPE]  # TODO slm.shape == image.shape?
    roi_res = (round(slm_res[0] * 0.8), round(slm_res[1] * 0.8))

    dtype = torch.float32  # default datatype (results may differ if using, e.g., float64)
    device = "cuda" if torch.cuda.is_available() else "cpu"  # TODO gpu is too small
    # device = "cpu"

    # Options for the algorithm
    lr_s_phase = lr_phase / 200
    loss_model = nn.MSELoss().to(
        device
    )  # loss function for SGD (or perceptualloss.PerceptualLoss())
    loss_phase = nn.MSELoss().to(device)
    loss_mse = nn.MSELoss().to(device)
    s0_phase = 1.0  # initial scale for phase optimization
    # s0_model = 1.0  # initial scale for model training
    sa = torch.tensor(s0_phase, device=device, requires_grad=True)
    sb = torch.tensor(0.3, device=device, requires_grad=True)

    num_iters_model_update = 1  # number of iterations for model-training subloops
    num_iters_phase_update = 1  # number of iterations for phase optimization

    # Path for data
    # result_path = "./models"
    # model_path = model_path  # path for new model checkpoints
    utils.cond_mkdir(model_path)
    # phase_path = phase_path  # path of precomputed phase pool
    # data_path = "./data"  # path of targets

    # Hardware setup
    camera_prop = PhysicalProp(
        slm_device,
        cam_device,
        slm_settle_time=slm_settle_time,
        # channel,
        # laser_arduino=True,
        # roi_res=(roi_res[1], roi_res[0]),  # TODO why inverted
        # range_row=(220, 1000),
        # range_col=(300, 1630),
        # patterns_path=calibration_path,  # path of 21 x 12 calibration patterns, see Supplement.
        # show_preview=True,
    )

    # Model instance to train
    # Check propagation_model.py for the default parameter settings!
    blur = utils.make_kernel_gaussian(0.85, 3)  # Optional, just be consistent with inference.
    model = ModelPropagate(
        distance=prop_dist,
        feature_size=feature_size,
        wavelength=wavelength,
        blur=blur,
        image_res=slm_res,  # TODO slm or image?
    ).to(device)

    if pretrained_path != "":
        print(f"   - Start from pre-trained model: {pretrained_path}")
        checkpoint = torch.load(pretrained_path)
        model.load_state_dict(checkpoint)
    model = model.train()

    # Augmented image loader (If you want to shuffle, augment dataset, put options accordingly)
    image_loader = ImageLoader(
        train_data_path,
        channel=channel,
        batch_size=batch_size,
        image_res=image_res,
        homography_res=roi_res,
        crop_to_homography=False,
        shuffle=True,
        vertical_flips=False,
        horizontal_flips=False,
    )

    # optimizer for model training
    # Note that indeed, you can set lrs of each parameters different! (especially for Source Amplitude params)
    # But it works well with the same lr.
    optimizer_model = optim.Adam(
        [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if "source_amp" not in name and "process_phase" not in name
                ]
            },
            {"params": model.source_amp.parameters(), "lr": lr_model * 1},
            {"params": model.process_phase.parameters(), "lr": lr_model * 1},
        ],
        lr=lr_model,
    )

    optimizer_phase_scale = optim.Adam([sa, sb], lr=lr_s_phase)
    if step_lr:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer_model, step_size=5, gamma=0.2
        )  # 1/5 every 3 epoch

    # tensorboard writer
    summaries_dir = os.path.join("runs", run_id)
    utils.cond_mkdir(summaries_dir)
    writer = SummaryModelWriter(
        model, f"{summaries_dir}", slm_res=slm_res, roi_res=roi_res, ch=channel
    )

    i_acc = 0
    for e in range(num_epochs):

        print(f"   - Epoch {e+1} ...")
        # visualize all the modules in the model on tensorboard
        with torch.no_grad():
            writer.visualize_model(e)

        for i, target in enumerate(image_loader):
            target_amp, _, target_filenames = target

            # extract indices of images
            idxs = []
            for name in target_filenames:
                _, target_filename = os.path.split(name)
                idxs.append(target_filename.split("_")[-1])
            target_amp = utils.crop_image(
                target_amp, target_shape=roi_res, stacked_complex=False
            ).to(device)

            # load phases
            slm_phases = []
            for k, idx in enumerate(idxs):
                # Load pre-computed phases
                # Instead, you can optimize phases from the scratch after a few number of iterations.
                if e > 0:
                    phase_filename = os.path.join(phase_path, f"{chan_str}", f"{idx}.png")
                else:
                    phase_filename = os.path.join(
                        phase_path, f"{chan_str}", f"{idx}_{channel}", "phasemaps_1000.png",
                    )

                if os.path.exists(phase_filename):
                    slm_phase = skimage.io.imread(phase_filename) / np.iinfo(np.uint8).max
                else:
                    slm_phase = (
                        np.random.randint(
                            low=0, high=np.iinfo(np.uint8).max + 1, size=(1, 1, *slm_res),
                        )
                        / np.iinfo(np.uint8).max
                    )

                # invert phase (our SLM setup) #TODO not needed to invert in our setting
                slm_phase = (
                    torch.tensor((1 - slm_phase) * 2 * np.pi - np.pi, dtype=dtype)
                    .reshape(1, 1, *slm_res)
                    .to(device)
                )
                slm_phases.append(slm_phase)
            slm_phases = torch.cat(slm_phases, 0).detach().requires_grad_(True)

            # optimizer for phase
            optimizer_phase = optim.Adam([slm_phases], lr=lr_phase)

            # 1) phase update loop
            model = model.eval()
            for _ in range(max(e * num_iters_phase_update, 1)):
                optimizer_phase.zero_grad()

                # propagate forward through the model
                recon_field = model(slm_phases)
                recon_amp = recon_field.abs()
                model_amp = utils.crop_image(
                    recon_amp, target_shape=roi_res, pytorch=True, stacked_complex=False
                )

                # calculate loss and backpropagate to phase
                with torch.no_grad():
                    scale_phase = (model_amp * target_amp).mean(dim=[-2, -1], keepdims=True) / (
                        model_amp ** 2
                    ).mean(dim=[-2, -1], keepdims=True)

                    # or we can optimize scale with regression and statistics of the image
                    # scale_phase = target_amp.mean(dim=[-2,-1], keepdims=True).detach() * sa + sb

                loss_value_phase = loss_phase(scale_phase * model_amp, target_amp)
                loss_value_phase.backward()
                optimizer_phase.step()
                optimizer_phase_scale.step()

            # write phase (update phase pool)
            with torch.no_grad():
                for k, idx in enumerate(idxs):
                    phase_out_8bit = utils.phasemap_8bit(
                        slm_phases[k, np.newaxis, ...].cpu().detach(), inverted=True
                    )
                    cv2.imwrite(os.path.join(phase_path, f"{idx}.png"), phase_out_8bit)

            # make slm phases 8bit variable as displayed
            slm_phases = utils.quantized_phase(slm_phases)

            # 2) display and capture
            camera_amp = []
            with torch.no_grad():
                # forward physical pass (display), capture and stack them in batch dimension
                for k, idx in enumerate(idxs):
                    slm_phase = slm_phases[k, np.newaxis, ...]
                    camera_amp.append(camera_prop(slm_phase))
                camera_amp = torch.cat(camera_amp, 0)

            # 3) model update loop
            model = model.train()
            for _ in range(num_iters_model_update):

                # zero grad
                optimizer_model.zero_grad()

                # propagate forward through the model
                recon_field = model(slm_phases)
                recon_amp = recon_field.abs()
                model_amp = utils.crop_image(
                    recon_amp, target_shape=roi_res, pytorch=True, stacked_complex=False
                )

                # ---------------------------------
                # TODO check rescaling
                camera_amp_scaled = torch.empty(
                    (camera_amp.shape[0], camera_amp.shape[1], slm_res[0], slm_res[1]),
                    device=device,
                )

                for p in range(camera_amp.shape[0]):
                    for c in range(camera_amp.shape[1]):
                        im = Image.fromarray(camera_amp[p, c].cpu().numpy())
                        im = im.resize(
                            (slm_res[1], slm_res[0]), Image.BICUBIC,  # Pillow uses width, height
                        )
                        camera_amp_scaled[p, c] = torch.from_numpy(np.array(im)).to(device)

                camera_amp = utils.crop_image(
                    camera_amp_scaled, target_shape=roi_res, pytorch=True, stacked_complex=False,
                )
                # ---------------------------------

                # calculate loss and backpropagate to model parameters
                loss_value_model = loss_model(model_amp, camera_amp)
                loss_value_model.backward()
                optimizer_model.step()

            # write to tensorboard #TODO WARNING:root:NaN or Inf found in input tensor.
            with torch.no_grad():
                if i % 50 == 0:
                    writer.add_scalar("Scale/sa", sa, i_acc)
                    writer.add_scalar("Scale/sb", sb, i_acc)
                    for idx_s in range(batch_size):
                        writer.add_scalar(
                            f"Scale/model_vs_target_{idx_s}", scale_phase[idx_s], i_acc
                        )
                    writer.add_scalar("Loss/model_vs_target", loss_value_phase, i_acc)
                    writer.add_scalar("Loss/model_vs_camera", loss_value_model, i_acc)
                    writer.add_scalar(
                        "Loss/camera_vs_target",
                        loss_mse(camera_amp * target_amp.mean() / camera_amp.mean(), target_amp,),
                        i_acc,
                    )
                if i % 50 == 0:
                    recon = model_amp[0, ...]
                    captured = camera_amp[0, ...]
                    gt = target_amp[0, ...] / scale_phase[0, ...]
                    max_amp = max(recon.max(), captured.max(), gt.max())
                    writer.add_image("Amp/recon", recon / max_amp, i_acc)
                    writer.add_image("Amp/captured", captured / max_amp, i_acc)
                    writer.add_image("Amp/target", gt / max_amp, i_acc)

                i_acc += 1

        # save model, every epoch
        torch.save(model.state_dict(), os.path.join(model_path, f"{run_id}_{e}epoch.pth"))
        if step_lr:
            lr_scheduler.step()

    # # disconnect everything #TODO needed? Add to camera?
    # if camera_prop is not None:
    #     camera_prop.disconnect()
    #     camera_prop.alc.disconnect()
