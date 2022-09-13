import math

import numpy as np
import torch

from mask_designer.utils import extend_to_field
from mask_designer.wrapper import fftshift, ifftshift, polar_to_rect
from mask_designer.experimental_setup import amp_mask


def __compute_H(prop_distance, wavelength, slm_shape, pixel_pitch):
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

    :param prop_distance: The propagation distance from the SLM to the target plane
    :type prop_distance: float
    :param wavelength: The wavelength of the light
    :type wavelength: float
    :param slm_shape: The shape or the resolution of the SLM
    :type slm_shape: tuple(int)
    :param pixel_pitch: The pixel pitch of the SLM
    :type pixel_pitch: float
    :return: H matrix
    :rtype: torch.Tensor
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
    H_exp = torch.mul(H_exp, prop_distance)

    # band-limited ASM - Matsushima et al. (2009)
    fy_max = 1 / np.sqrt((2 * prop_distance * (1 / y)) ** 2 + 1) / wavelength
    fx_max = 1 / np.sqrt((2 * prop_distance * (1 / x)) ** 2 + 1) / wavelength
    H_filter = torch.tensor(
        ((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=torch.float32,
    )

    # get real/img components
    H_real, H_imag = polar_to_rect(H_filter, H_exp)

    H = torch.stack((H_real, H_imag), 4)
    H = ifftshift(H)
    H = torch.view_as_complex(H)

    return H


def holoeye_lens_to_lensless(
    holoeye_field, prop_distance, wavelength, slm_shape, pixel_pitch
):  # TODO name
    """
    Transform from normal setting (with lens) to the lensless setting used by neural
    holography.

    :param holoeye_field: The field that needs to be transformed
    :type holoeye_field: torch.Tensor
    :param prop_distance: The propagation distance from the SLM to the target plane
    :type prop_distance: float
    :param wavelength: The wavelength of the light
    :type wavelength: float
    :param slm_shape: The shape or the resolution of the SLM
    :type slm_shape: tuple(int)
    :param pixel_pitch: The pixel pitch of the SLM
    :type pixel_pitch: float
    :return: torch.Tensor
    :rtype: The transformed field
    """
    H = __compute_H(prop_distance, wavelength, slm_shape, pixel_pitch)

    angles = holoeye_field.angle()
    field = torch.polar(amp_mask, angles)

    field = fftshift(
        torch.fft.ifftn(
            torch.fft.fftn(
                torch.fft.fftn(field, dim=(-2, -1), norm="ortho"), dim=(-2, -1), norm="ortho",
            )
            / H,
            dim=(-2, -1),
            norm="ortho",
        )
    )

    return extend_to_field(field.angle())


def neural_holography_lensless_to_lens(  # TODO name
    neural_holography_field, prop_distance, wavelength, slm_shape, pixel_pitch
):
    """
    Transform from the lensless setting used by neural holography to the the
    normal lens setting.

    :param neural_holography_field: The field that needs to be transformed
    :type neural_holography_field: torch.Tensor
    :param prop_distance: The propagation distance from the SLM to the target plane
    :type prop_distance: float
    :param wavelength: The wavelength of the light
    :type wavelength: float
    :param slm_shape: The shape or the resolution of the SLM
    :type slm_shape: tuple(int)
    :param pixel_pitch: The pixel pitch of the SLM
    :type pixel_pitch: float
    :return: torch.Tensor
    :rtype: The transformed field
    """
    H = __compute_H(prop_distance, wavelength, slm_shape, pixel_pitch)

    angles = neural_holography_field.angle()
    field = torch.polar(amp_mask, angles)

    field = torch.fft.ifftn(
        torch.fft.ifftn(
            H * torch.fft.fftn(ifftshift(field), dim=(-2, -1), norm="ortho"),
            dim=(-2, -1),
            norm="ortho",
        ),
        dim=(-2, -1),
        norm="ortho",
    )

    return extend_to_field(field.angle())
