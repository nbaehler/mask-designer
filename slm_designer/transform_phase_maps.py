import torch
import numpy as np
import math

from slm_designer.wrapper import fftshift, ifftshift, polar_to_rect


def __compute_H(prop_dist, wavelength, slm_shape, pixel_pitch):
    """
    https://github.com/computational-imaging/neural-holography/blob/d2e399014aa80844edffd98bca34d2df80a69c84/propagation_ASM.py

    Compute H which is used in neural holography code, core is imported as is.

    This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
        # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
        # The material is provided as-is, with no warranties whatsoever.
        # If you publish any code, data, or scientific work based on this, please cite our work.

    Technical Paper:
    Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.

    Copyright (c) 2020, Stanford University

    All rights reserved.

    Refer to the LICENSE file for more information.

    Parameters
    ----------
    prop_dist : float
        The propagation distance from the SLM to the target plane
    wavelength : float
        The wavelength of the light
    slm_shape : tuple(int)
        The shape or the resolution of the SLM
    pixel_pitch : float
        The pixel pitch of the SLM

    Returns
    -------
    torch.Tensor
        H, the homography matrix
    """

    # number of pixels
    num_y, num_x = slm_shape

    # sampling interval size
    dy, dx = pixel_pitch

    # size of the field
    y, x = (dy * float(num_y), dx * float(num_x))

    # frequency coordinates sampling
    fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y)
    fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x)

    # momentum/reciprocal space
    FX, FY = np.meshgrid(fx, fy)

    # transfer function in numpy (omit distance)
    HH = 2 * math.pi * np.sqrt(1 / wavelength ** 2 - (FX ** 2 + FY ** 2))

    # create tensor & upload to device (GPU)
    H_exp = torch.tensor(HH, dtype=torch.float32)

    # reshape tensor and multiply
    H_exp = torch.reshape(H_exp, (1, 1, *H_exp.size()))

    # multiply by distance
    H_exp = torch.mul(H_exp, prop_dist)

    # band-limited ASM - Matsushima et al. (2009)
    fy_max = 1 / np.sqrt((2 * prop_dist * (1 / y)) ** 2 + 1) / wavelength
    fx_max = 1 / np.sqrt((2 * prop_dist * (1 / x)) ** 2 + 1) / wavelength
    H_filter = torch.tensor(
        ((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8),
        dtype=torch.float32,
    )

    # get real/img components
    H_real, H_imag = polar_to_rect(H_filter, H_exp)

    H = torch.stack((H_real, H_imag), 4)
    H = ifftshift(H)
    H = torch.view_as_complex(H)

    return H


def transform_to_neural_holography_setting(
    holoeye_phase_map, prop_dist, wavelength, slm_shape, pixel_pitch
):
    """
    Transform from normal setting (with lens) to the lensless setting used by neural
    holography.

    Parameters
    ----------
    holoeye_phase_map : torch.Tensor
        The phase map that needs to be transformed
    prop_dist : float
        The propagation distance from the SLM to the target plane
    wavelength : float
        The wavelength of the light
    slm_shape : tuple(int)
        The shape or the resolution of the SLM
    pixel_pitch : float
        The pixel pitch of the SLM

    Returns
    -------
    torch.Tensor
        The transformed phase map
    """

    H = __compute_H(prop_dist, wavelength, slm_shape, pixel_pitch)

    return fftshift(
        torch.fft.ifftn(
            torch.fft.fftn(
                torch.fft.fftn(holoeye_phase_map, dim=(-2, -1), norm="ortho"),
                dim=(-2, -1),
                norm="ortho",
            )
            / H,
            dim=(-2, -1),
            norm="ortho",
        )
    )


def transform_from_neural_holography_setting(
    neural_holography_phase_map, prop_dist, wavelength, slm_shape, pixel_pitch
):
    """
    Transform from the lensless setting used by neural holography to the the
    normal lens setting.

    Parameters
    ----------
    neural_holography_phase_map : torch.Tensor
        The phase map that needs to be transformed
    prop_dist : float
        The propagation distance from the SLM to the target plane
    wavelength : float
        The wavelength of the light
    slm_shape : tuple(int)
        The shape or the resolution of the SLM
    pixel_pitch : float
        The pixel pitch of the SLM

    Returns
    -------
    torch.Tensor
        The transformed phase map
    """
    H = __compute_H(prop_dist, wavelength, slm_shape, pixel_pitch)

    return torch.fft.ifftn(
        torch.fft.ifftn(
            H
            * torch.fft.fftn(
                ifftshift(neural_holography_phase_map), dim=(-2, -1), norm="ortho"
            ),
            dim=(-2, -1),
            norm="ortho",
        ),
        dim=(-2, -1),
        norm="ortho",
    )