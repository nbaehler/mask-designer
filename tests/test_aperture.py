import pytest
from slm_designer.slm import SLM
from slm_designer.aperture import rect_aperture


class TestAperture:
    """
    Test :py:module:`~slm_designer.aperture`.
    """

    def test_rect_aperture(self):
        slm_shape = (10, 10)
        pixel_pitch = (0.18e-3, 0.18e-3)
        apert_dim = (2 * pixel_pitch[0], 2 * pixel_pitch[1])

        # valid
        rect_aperture(slm_shape=slm_shape, pixel_pitch=pixel_pitch, apert_dim=apert_dim)

        # invalid, outside SLM
        slm = SLM(shape=slm_shape, pixel_pitch=pixel_pitch)
        with pytest.raises(AssertionError, match="must lie within SLM dimensions"):
            rect_aperture(
                slm_shape=slm_shape,
                pixel_pitch=pixel_pitch,
                apert_dim=apert_dim,
                center=(slm.height, slm.width),
            )

        # aperture extends beyond
        with pytest.raises(ValueError, match="extends past valid SLM dimensions"):
            rect_aperture(
                slm_shape=slm_shape,
                pixel_pitch=pixel_pitch,
                apert_dim=apert_dim,
                center=(slm.height - pixel_pitch[0], slm.width - pixel_pitch[1]),
            )
