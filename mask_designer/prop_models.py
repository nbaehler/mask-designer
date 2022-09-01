import torch
from waveprop.rs import angular_spectrum

from mask_designer.wrapper import ModelPropagate, PhysicalProp, propagation_ASM

# from enum import Enum

# # Camera devices that are implemented in this project
# class PropModel(Enum):
#     ASM = "ASM"
#     CAMERA = "CAMERA"
#     MODEL = "MODEL"
#     WAVEPROP = "WAVEPROP"


def propagator_waveprop_angular_spectrum(  # TODO buggy
    u_in,
    feature_size,
    wavelength,
    z,
    # linear_conv=True,
    # padtype="zero",
    return_H=False,
    precomped_H=None,
    return_H_exp=False,
    precomped_H_exp=None,
    dtype=torch.float32,
):
    if return_H or return_H_exp:
        return angular_spectrum(
            u_in=u_in[0, 0, :, :],
            wv=wavelength,
            d1=feature_size[0],
            dz=z,
            # linear_conv=True, # TODO check those two parameters
            # padtype="zero",
            return_H=return_H,
            H=precomped_H,
            return_H_exp=return_H_exp,
            H_exp=precomped_H_exp,
            dtype=dtype,
            device="cuda",  # TODO always on gpu?
        )

    res, _, _ = angular_spectrum(
        u_in=u_in[0, 0, :, :],
        wv=wavelength,
        d1=feature_size[0],
        dz=z,
        # linear_conv=True, # TODO check those two parameters
        # padtype="zero",
        return_H=False,
        H=precomped_H,
        return_H_exp=False,
        H_exp=precomped_H_exp,
        dtype=dtype,
        device="cuda",  # TODO always on gpu?
    )

    return res[None, None, :, :]


# prop_models = {
#     PropModel.ASM: propagation_ASM,
#     PropModel.CAMERA: PhysicalProp,
#     PropModel.MODEL: ModelPropagate,
#     PropModel.WAVEPROP: propagator_waveprop_angular_spectrum,
# }
