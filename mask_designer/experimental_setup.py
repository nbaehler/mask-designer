from enum import Enum

import torch
from skimage.draw import disk
from slm_controller.hardware import SLMDevices, SLMParam, slm_devices

from mask_designer.hardware import CamDevices


# Parameters relevant for the experiments
class Params(Enum):
    WAVELENGTH = "wavelength"
    PROPAGATION_DISTANCE = "prop_distance"
    SLM_SETTLE_TIME = "slm_settle_time"
    SLM_SHOW_TIME = "slm_show_time"
    ROI = "roi"

    @staticmethod
    def values():
        return [param.value for param in Params]


# Actual values of those parameters
params = {
    Params.WAVELENGTH: 532e-9,
    Params.PROPAGATION_DISTANCE: 0.315,
    Params.SLM_SETTLE_TIME: 0.25,
    Params.SLM_SHOW_TIME: 10,
    Params.ROI: (320, 560),
}

# Choose a slm device
slm_device = SLMDevices.HOLOEYE_LC_2012.value

# and a camera device that you want to use
cam_device = CamDevices.IDS.value


def circular_amp():  # TODO use circ aperture from our repo, documentation, from here https://stackoverflow.com/a/70283438
    # Laser radius = 1cm := r
    # pixel pitch = 0.36e-4 m = 0.0036 cm,
    # ==> r = 278 px

    shape = slm_devices[slm_device][SLMParam.SLM_SHAPE]

    amp_mask = torch.zeros(shape)

    center = (shape[0] / 2, shape[1] / 2)
    radius = 278
    rr, cc = disk(center, radius)
    amp_mask[rr, cc] = 1

    return amp_mask


def rectangular_amp():
    return torch.ones(slm_devices[slm_device][SLMParam.SLM_SHAPE])


# Chose the shape of the laser beam hitting the SLM
amp_mask = circular_amp()
