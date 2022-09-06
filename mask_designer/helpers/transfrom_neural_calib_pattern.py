import sys
from os.path import abspath, dirname, join

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, "../.."))
sys.path.append(CODE_DIR)


import numpy as np
from mask_designer.experimental_setup import Params, params, slm_device
from mask_designer.utils import (
    angularize_phase_mask,
    extend_to_field,
    quantize_phase_mask,
)
from PIL import Image
from slm_controller.hardware import SLMParam, slm_devices


def main():
    calib_phase_img = Image.open("citl/calibration/phase_mask.png")
    calib_phase = np.array(calib_phase_img)
    calib_phase = np.mean(calib_phase, axis=2)

    field = extend_to_field(angularize_phase_mask(calib_phase))

    prop_dist = params[Params.PROPAGATION_DISTANCE]
    wavelength = params[Params.WAVELENGTH]
    pixel_pitch = slm_devices[slm_device][SLMParam.PIXEL_PITCH]
    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]

    from mask_designer.transform_fields import (
        transform_to_neural_holography_setting,
    )  # TODO move import up!!

    # Transform the results to the hardware setting using a lens
    field = transform_to_neural_holography_setting(
        field, prop_dist, wavelength, slm_shape, pixel_pitch
    )

    phase_mask = quantize_phase_mask(field.angle())

    phase_mask_img = Image.fromarray(phase_mask)
    phase_mask_img.save("citl/calibration/green.png")


if __name__ == "__main__":
    main()
