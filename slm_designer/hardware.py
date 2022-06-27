from enum import Enum

# Camera devices that are implemented in this project
class CamDevices(Enum):
    DUMMY = "dummy"  # TODO for development only!
    IDS = "ids"

    @staticmethod
    def values():
        return [device.value for device in CamDevices]


# Parameters of those cameras
class CamParam:
    IMG_SHAPE = "img_shape"


# Actual values of those parameters for all the cameras
cam_devices = {
    CamDevices.DUMMY.value: {CamParam.IMG_SHAPE: (1216, 1936)},
    CamDevices.IDS.value: {CamParam.IMG_SHAPE: (1216, 1936)},
}
