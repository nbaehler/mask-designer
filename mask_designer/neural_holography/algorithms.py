"""
https://github.com/computational-imaging/neural-holography/blob/d2e399014aa80844edffd98bca34d2df80a69c84/algorithms.py

This is the algorithm script used for the representative iterative CGH implementations, i.e., GS and SGD.

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

import mask_designer.neural_holography.utils as utils
import torch
import torch.nn as nn
import torch.optim as optim
from mask_designer.experimental_setup import amp_mask
from mask_designer.neural_holography.prop_asm import prop_asm


# 1. GS
def gerchberg_saxton(
    init_phase,
    target_amp,
    num_iters,
    prop_dist,
    wavelength,
    feature_size=6.4e-6,
    prop_model="ASM",
    propagator=prop_asm,
    dtype=torch.float32,
    precomputed_H_f=None,
    precomputed_H_b=None,
):
    """
    Given the initial guess, run the SGD algorithm to calculate the optimal phase mask of spatial light modulator

    :param init_phase: a tensor, in the shape of (1,1,H,W), initial guess for the phase.
    :param target_amp: a tensor, in the shape of (1,1,H,W), the amplitude of the target image.
    :param num_iters: the number of iterations to run the GS.
    :param prop_dist: propagation distance in m.
    :param wavelength: wavelength in m.
    :param feature_size: the SLM pixel pitch, in meters, default 6.4e-6
    :param prop_model: string indicating the light transport model, default 'ASM'. ex) 'ASM', 'fresnel', 'model'
    :param propagator: predefined function or model instance for the propagation.
    :param writer: tensorboard writer
    :param dtype: torch datatype for computation at different precision, default torch.float32.
    :param precomputed_H_f: A Pytorch complex64 tensor, pre-computed kernel for forward prop (SLM to image)
    :param precomputed_H_b: A Pytorch complex64 tensor, pre-computed kernel for backward propagation (image to SLM)

    Output
    ------
    :return: a tensor, the optimized phase mask at the SLM plane, in the shape of (1,1,H,W)
    """

    slm_amp = amp_mask.to(init_phase.device)

    # initial guess; random phase
    real, imag = utils.polar_to_rect(slm_amp, init_phase)
    slm_field = torch.complex(real, imag)

    # run the GS algorithm
    for _ in range(num_iters):
        # SLM plane to image plane
        recon_field = utils.propagate_field(
            slm_field,
            propagator,
            prop_dist,
            wavelength,
            feature_size,
            prop_model,
            dtype,
            precomputed_H_f,
        )

        # replace amplitude at the image plane
        recon_field = utils.replace_amplitude(recon_field, target_amp)

        # image plane to SLM plane
        slm_field = utils.propagate_field(
            recon_field,
            propagator,
            -prop_dist,
            wavelength,
            feature_size,
            prop_model,
            dtype,
            precomputed_H_b,
        )

        # amplitude constraint at the SLM plane
        slm_field = utils.replace_amplitude(slm_field, slm_amp)

    # return phases
    return slm_field.angle()


# 2. SGD
def stochastic_gradient_descent(
    init_phase,
    target_amp,
    num_iters,
    prop_dist,
    wavelength,
    feature_size,
    roi_res=None,
    prop_model="ASM",
    propagator=prop_asm,
    loss=nn.MSELoss(),
    lr=0.01,
    lr_s=0.003,
    s0=1.0,
    writer=None,
    dtype=torch.float32,
    precomputed_H=None,
):

    """
    Given the initial guess, run the SGD algorithm to calculate the optimal phase mask of spatial light modulator.

    Input
    ------
    :param init_phase: a tensor, in the shape of (1,1,H,W), initial guess for the phase.
    :param target_amp: a tensor, in the shape of (1,1,H,W), the amplitude of the target image.
    :param num_iters: the number of iterations to run the SGD.
    :param prop_dist: propagation distance in m.
    :param wavelength: wavelength in m.
    :param feature_size: the SLM pixel pitch, in meters, default 6.4e-6
    :param roi_res: a tuple of integer, region of interest, like (880, 1600)
    :param prop_model: a string, that indicates the propagation model. ('ASM' or 'MODEL')
    :param propagator: predefined function or model instance for the propagation.
    :param loss: loss function, default L2
    :param lr: learning rate for optimization variables
    :param lr_s: learning rate for learnable scale
    :param s0: initial scale
    :param writer: Tensorboard writer instance
    :param dtype: default torch.float32
    :param precomputed_H: A Pytorch complex64 tensor, pre-computed kernel shape of (1,1,2H,2W) for fast computation.

    Output
    ------
    :return: a tensor, the optimized phase mask at the SLM plane, in the shape of (1,1,H,W)
    """

    device = init_phase.device
    s = torch.tensor(s0, requires_grad=True, device=device)

    # phase at the slm plane
    slm_phase = init_phase.requires_grad_(True)

    temp = slm_phase.clone().detach()  # TODO remove

    # optimization variables and adam optimizer
    optimizer = optim.Adam([slm_phase], lr=lr)

    if lr_s > 0:
        optimizer.add_param_group({"params": s, "lr": lr_s})

    # crop target roi
    target_amp = utils.crop_image(target_amp, roi_res, stacked_complex=False)

    # Only needed for the simulation
    slm_amp = amp_mask.to(device)[None, None, :, :]

    print("Requires grad", slm_phase.requires_grad)

    # run the iterative algorithm
    for k in range(num_iters):
        optimizer.zero_grad()

        print(f"Diff phase {str(torch.sum((temp - slm_phase)**2).item())}")  # TODO remove

        print(f"Same {str(temp.equal(slm_phase))}")  # TODO remove

        print(f"Scale {str(s.item())}")  # TODO remove

        print("Requires grad", slm_phase.requires_grad)

        # forward propagation from the SLM plane to the target plane
        real, imag = utils.polar_to_rect(slm_amp, slm_phase)
        slm_field = torch.complex(real, imag)

        recon_field = utils.propagate_field(
            slm_field,
            propagator,
            prop_dist,
            wavelength,
            feature_size,
            prop_model,
            dtype,
            precomputed_H,
        )

        # get amplitude
        recon_amp = recon_field.abs()  # TODO not in [0,1], amp can get bigger, normalize?

        # crop roi
        recon_amp = utils.crop_image(recon_amp, target_shape=roi_res, stacked_complex=False)

        print(  # TODO remove
            "recon_amp",
            torch.min(recon_amp).item(),
            torch.max(recon_amp).item(),
            torch.median(recon_amp).item(),
            torch.mean(recon_amp).item(),
            torch.quantile(recon_amp, 0.99).item(),
        )

        # camera-in-the-loop technique
        if prop_model.upper() == "PHYSICAL":
            print("Requires grad", slm_phase.requires_grad)

            captured_amp = propagator(slm_phase)  # .detach()

            print("Requires grad", slm_phase.requires_grad)

            print(  # TODO remove
                "captured_amp",
                torch.min(captured_amp).item(),
                torch.max(captured_amp).item(),
                torch.median(captured_amp).item(),
                torch.mean(captured_amp).item(),
                torch.quantile(captured_amp, 0.99).item(),
            )

            # use the gradient of proxy, replacing the amplitudes
            # captured_amp is assumed that its size already matches that of recon_amp
            out_amp = recon_amp + (captured_amp - recon_amp).detach()
            # out_amp = captured_amp.detach()  # TODO not enough?
        else:
            out_amp = recon_amp

        # -----------------------------------------------------------------

        # # camera-in-the-loop technique # TODO this version is better and should work
        # if prop_model.upper() == "PHYSICAL":
        #     print("Requires grad", slm_phase.requires_grad)

        #     out_amp = propagator(slm_phase)  # .detach()

        #     print("Requires grad", slm_phase.requires_grad)

        #     print(  # TODO remove
        #         "out_amp",
        #         torch.min(out_amp).item(),
        #         torch.max(out_amp).item(),
        #         torch.median(out_amp).item(),
        #         torch.mean(out_amp).item(),
        #         torch.quantile(out_amp, 0.99).item(),
        #     )
        # else:
        #     print("Requires grad", slm_phase.requires_grad)

        #     # forward propagation from the SLM plane to the target plane
        #     real, imag = utils.polar_to_rect(slm_amp, slm_phase)
        #     slm_field = torch.complex(real, imag)

        #     recon_field = utils.propagate_field(
        #         slm_field,
        #         propagator,
        #         prop_dist,
        #         wavelength,
        #         feature_size,
        #         prop_model,
        #         dtype,
        #         precomputed_H,
        #     )

        #     # get amplitude
        #     recon_amp = recon_field.abs()  # TODO not in [0,1], amp can get bigger, normalize?

        #     # crop roi
        #     recon_amp = utils.crop_image(recon_amp, target_shape=roi_res, stacked_complex=False)

        #     print(  # TODO remove
        #         "recon_amp",
        #         torch.min(recon_amp).item(),
        #         torch.max(recon_amp).item(),
        #         torch.median(recon_amp).item(),
        #         torch.mean(recon_amp).item(),
        #         torch.quantile(recon_amp, 0.99).item(),
        #     )

        #     out_amp = recon_amp

        print(  # TODO remove
            "out_amp",
            torch.min(out_amp).item(),
            torch.max(out_amp).item(),
            torch.median(out_amp).item(),
            torch.mean(out_amp).item(),
            torch.quantile(out_amp, 0.99).item(),
        )

        print(  # TODO remove
            "s * out_amp",
            torch.min(s * out_amp).item(),
            torch.max(s * out_amp).item(),
            torch.median(s * out_amp).item(),
            torch.mean(s * out_amp).item(),
            torch.quantile(s * out_amp, 0.99).item(),
        )

        print(  # TODO remove
            "target_amp",
            torch.min(target_amp).item(),
            torch.max(target_amp).item(),
            torch.median(target_amp).item(),
            torch.mean(target_amp).item(),
            torch.quantile(target_amp, 0.99).item(),
        )

        # calculate loss and backprop
        lossValue = loss(s * out_amp, target_amp)

        print(f"Loss {(lossValue.item())}")  # TODO remove

        print("Is leaf", slm_phase.is_leaf)  # TODO remove

        # slm_phase.retain_grad()
        # slm_phase.requires_grad_(True)

        # slm_phase.register_hook(
        #     lambda d: print("Phase grad non zero", (torch.count_nonzero(d) > 0).item())
        # )

        print("Requires grad", slm_phase.requires_grad)

        lossValue.backward()

        print("Phase grad non zero", (torch.count_nonzero(slm_phase.grad.data) > 0).item())
        print("Scale grad", s.grad.data.item())

        optimizer.step()

        from mask_designer.utils import save_image, quantize_phase_mask  # TODO remove
        import datetime

        name = str(datetime.datetime.now().time()).replace(":", "_").replace(".", "_")
        save_image(
            quantize_phase_mask(slm_phase.cpu().detach().numpy()[0, 0, :, :]),
            f"citl/snapshots/phase_{name}.png",
        )

        # write to tensorboard / write phase image
        # Note that it takes 0.~ s for writing it to tensorboard
        with torch.no_grad():
            if k % 50 == 0:
                utils.write_sgd_summary(
                    out_amp, target_amp, k, writer=writer, s=s, prefix="test",
                )

    # print(torch.max(slm_phase), torch.min(slm_phase)) # TODO not in [-pi, pi]??

    return slm_phase
