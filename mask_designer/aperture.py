"""
author: Eric Bezzam,
email: ebezzam@gmail.com,
GitHub: https://github.com/ebezzam
"""


from enum import Enum

import numpy as np

from mask_designer.virtual_slm import VirtualSLM


class ApertureOptions(Enum):
    RECT = "rect"
    SQUARE = "square"
    LINE = "line"
    CIRC = "circ"

    @staticmethod
    def values():
        return [shape.value for shape in ApertureOptions]


def rect_aperture(slm_shape, pixel_pitch, apert_dim, center=None):
    """
    Create and return VirtualSLM object with rectangular aperture of desired dimensions.

    :param slm_shape: Dimensions (height, width) of VirtualSLM in cells.
    :type slm_shape: tuple(int)
    :param pixel_pitch: Dimensions (height, width) of each cell in meters.
    :type pixel_pitch: tuple(float)
    :param apert_dim: Dimensions (height, width) of aperture in meters.
    :type apert_dim: tuple(float)
    :param center: [Optional] center of aperture along (SLM) coordinates, indexing starts in top-left corner.
        Default is to place center of aperture at center of SLM, defaults to
        None # TODO improve
    :type center: tuple(float), optional
    :raises ValueError: _description_ # TODO: add description
    :return: VirtualSLM object with cells programmed to desired rectangular aperture.
    :rtype: :py:class:`~mask_designer.virtual_slm.VirtualSLM`
    """
    # check input values
    assert np.all(apert_dim) > 0

    # initialize SLM
    slm = VirtualSLM(shape=slm_shape, pixel_pitch=pixel_pitch)

    # check / compute center
    if center is None:
        center = slm.center
    else:
        assert (
            0 <= center[0] < slm.height
        ), f"Center {center} must lie within VirtualSLM dimensions {slm.dim}."
        assert (
            0 <= center[1] < slm.width
        ), f"Center {center} must lie within VirtualSLM dimensions {slm.dim}."

    # compute mask
    apert_dim = np.array(apert_dim)
    top_left = center - apert_dim / 2
    bottom_right = top_left + apert_dim
    if (
        top_left[0] < 0
        or top_left[1] < 0
        or bottom_right[0] >= slm.dim[0]
        or bottom_right[1] >= slm.dim[1]
    ):
        raise ValueError(
            f"Aperture ({top_left[0]}:{bottom_right[0]}, "
            f"{top_left[1]}:{bottom_right[1]}) extends past valid "
            f"VirtualSLM dimensions {slm.dim}"
        )
    slm.at(
        physical_coord=np.s_[top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]],
        value=255,
    )

    return slm


def line_aperture(slm_shape, pixel_pitch, length, vertical=True, center=None):
    """
    Create and return VirtualSLM object with a line aperture of desired length.


    :param slm_shape: Dimensions (height, width) of VirtualSLM in cells.
    :type slm_shape: tuple(int)
    :param pixel_pitch: Dimensions (height, width) of each cell in meters.
    :type pixel_pitch: tuple(float)
    :param length: Length of aperture in meters.
    :type length: float
    :param vertical: _description_, defaults to True # TODO: add description
    :type vertical: bool, optional
    :param center: [Optional] center of aperture along (SLM) coordinates, indexing starts in top-left corner.
        Default is to place center of aperture at center of SLM., defaults to
        None # TODO improve
    :type center: tuple(float), optional
    :return: VirtualSLM object with cells programmed to desired line aperture.
    :rtype: :py:class:`~mask_designer.virtual_slm.VirtualSLM`
    """
    # call `create_rect_aperture`
    apert_dim = (length, pixel_pitch[1]) if vertical else (pixel_pitch[0], length)
    return rect_aperture(slm_shape, pixel_pitch, apert_dim, center)


def square_aperture(slm_shape, pixel_pitch, side, center=None):
    """
    Create and return VirtualSLM object with a square aperture of desired shape.

    :param slm_shape: Dimensions (height, width) of VirtualSLM in cells.
    :type slm_shape: tuple(int)
    :param pixel_pitch: Dimensions (height, width) of each cell in meters.
    :type pixel_pitch: tuple(float)
    :param side: Side length of square aperture in meters.
    :type side: float
    :param center: [Optional] center of aperture along (SLM) coordinates, indexing starts in top-left corner.
        Default is to place center of aperture at center of SLM., defaults to
        None # TODO improve
    :type center: tuple(float), optional
    :return: VirtualSLM object with cells programmed to desired square aperture.
    :rtype: :py:class:`~mask_designer.virtual_slm.VirtualSLM`
    """
    return rect_aperture(slm_shape, pixel_pitch, (side, side), center)


def circ_aperture(slm_shape, pixel_pitch, radius, center=None):
    """
    Create and return VirtualSLM object with a circle aperture of desired shape.

    :param slm_shape: Dimensions (height, width) of VirtualSLM in cells.
    :type slm_shape: tuple(int)
    :param pixel_pitch: Dimensions (height, width) of each cell in meters.
    :type pixel_pitch: tuple(float)
    :param radius: Radius of aperture in meters.
    :type radius: float
    :param center: [Optional] center of aperture along (SLM) coordinates, indexing starts in top-left corner.
        Default is to place center of aperture at center of SLM., defaults to
        None # TODO improve
    :type center: tuple(float), optional
    :return: VirtualSLM object with cells programmed to desired circle aperture.
    :rtype: :py:class:`~mask_designer.virtual_slm.VirtualSLM`
    """
    # check input values
    assert radius > 0

    # initialize SLM
    slm = VirtualSLM(shape=slm_shape, pixel_pitch=pixel_pitch)

    # check / compute center
    if center is None:
        center = slm.center
    else:
        assert (
            0 <= center[0] < slm.height
        ), f"Center {center} must lie within VirtualSLM dimensions {slm.dim}."
        assert (
            0 <= center[1] < slm.width
        ), f"Center {center} must lie within VirtualSLM dimensions {slm.dim}."

    # compute mask
    i, j = np.meshgrid(
        np.arange(slm.dim[0], step=slm.pixel_pitch[0]),
        np.arange(slm.dim[1], step=slm.pixel_pitch[1]),
        sparse=True,
        indexing="ij",
    )
    x2 = (i - center[0]) ** 2
    y2 = (j - center[1]) ** 2
    slm[:] = 255 * (x2 + y2 < radius ** 2)
    return slm
