from enum import Enum


# Camera devices that are implemented in this project
class CamDevices(Enum):
    DUMMY = "dummy"
    IDS = "ids"

    @staticmethod
    def values():
        return [device.value for device in CamDevices]


# Parameters of those cameras
class CamParam:
    SHAPE = "shape"


# Actual values of those parameters for all the cameras
cam_devices = {
    CamDevices.DUMMY.value: {CamParam.SHAPE: (1216, 1936)},
    CamDevices.IDS.value: {CamParam.SHAPE: (1216, 1936)},
}
