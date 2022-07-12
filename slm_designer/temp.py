import numpy as np
from PIL import Image

from slm_designer.experimental_setup import slm_device

from slm_controller.hardware import (
    SLMParam,
    slm_devices,
)

from slm_designer.utils import resize_image_to_shape


def temp():
    slm_shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]

    image = Image.open("images/target_amplitude/holoeye_logo.png")
    image = np.array(image)
    images = [image]

    resized_image = resize_image_to_shape(images, slm_shape, pad=True)[0]
    resized_image = Image.fromarray(resized_image)

    resized_image.save("images/test/holoeye_logo_big.png")


if __name__ == "__main__":
    temp()
