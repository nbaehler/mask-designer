from slm_controller.hardware import SLMDevices
from slm_designer.hardware import CamDevices

from enum import Enum

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
    Params.PROPAGATION_DISTANCE: 0.34,
    # Params.SLM_SETTLE_TIME: 7,
    # Params.SLM_SHOW_TIME: 0.5,
    Params.SLM_SETTLE_TIME: 0.25,
    Params.SLM_SHOW_TIME: 1,
    # Params.ROI: (640, 880),
    Params.ROI: (320, 560),
}

# Choose a slm device
slm_device = (
    SLMDevices.HOLOEYE_LC_2012.value
)  # TODO does this structure still make sense?

# and a camera device that you want to use
cam_device = CamDevices.IDS.value
# cam_device = CamDevices.DUMMY.value
