from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)


import numpy as np
from PIL import Image

from slm_controller.hardware import (
    SLMParam,
    slm_devices,
)

from mask_designer.experimental_setup import Params, params, slm_device
from mask_designer.utils import (
    angularize_phase_mask,
    build_field,
    quantize_phase_mask,
)


def main():
    calib_phase_img = Image.open("citl/calibration/phase_mask.png")
    calib_phase = np.array(calib_phase_img)
    calib_phase = np.mean(calib_phase, axis=2)

    field = build_field(
        angularize_phase_mask(calib_phase)
    )  # TODO angularize and quantize should be inverses of one another, test that!
    # Check the conversions and division you do!

    prop_dist = params[Params.PROPAGATION_DISTANCE]
    wavelength = params[Params.WAVELENGTH]
    pixel_pitch = slm_devices[slm_device][SLMParam.PIXEL_PITCH]
    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]

    from mask_designer.transform_fields import transform_to_neural_holography_setting

    # Transform the results to the hardware setting using a lens
    field = transform_to_neural_holography_setting(
        field, prop_dist, wavelength, slm_shape, pixel_pitch
    )

    phase_mask = quantize_phase_mask(field.angle())

    phase_mask_img = Image.fromarray(phase_mask)
    phase_mask_img.save("citl/calibration/green.png")


if __name__ == "__main__":
    main()