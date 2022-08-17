"""
author: Eric Bezzam,
email: ebezzam@gmail.com,
GitHub: https://github.com/ebezzam
"""

from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, ".."))
sys.path.append(CODE_DIR)

import pytest
from mask_designer.virtual_slm import VirtualSLM
from mask_designer.aperture import rect_aperture


class TestAperture:
    """
    Test :py:module:`~mask_designer.aperture`.
    """

    def test_rect_aperture(self):
        slm_shape = (10, 10)
        pixel_pitch = (0.18e-3, 0.18e-3)
        apert_dim = (2 * pixel_pitch[0], 2 * pixel_pitch[1])

        # valid
        rect_aperture(slm_shape=slm_shape, pixel_pitch=pixel_pitch, apert_dim=apert_dim)

        # invalid, outside SLM
        virtual_slm = VirtualSLM(shape=slm_shape, pixel_pitch=pixel_pitch)
        with pytest.raises(AssertionError, match="must lie within VirtualSLM dimensions"):
            rect_aperture(
                slm_shape=slm_shape,
                pixel_pitch=pixel_pitch,
                apert_dim=apert_dim,
                center=(virtual_slm.height, virtual_slm.width),
            )

        # aperture extends beyond
        with pytest.raises(ValueError, match="extends past valid VirtualSLM dimensions"):
            rect_aperture(
                slm_shape=slm_shape,
                pixel_pitch=pixel_pitch,
                apert_dim=apert_dim,
                center=(virtual_slm.height - pixel_pitch[0], virtual_slm.width - pixel_pitch[1],),
            )
