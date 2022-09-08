import torch
from waveprop.rs import angular_spectrum

from mask_designer.transform_fields import neural_holography_lensless_to_lens


def prop_waveprop_asm(
    u_in,
    feature_size,
    wavelength,
    z,
    # linear_conv=True, # TODO check those two parameters
    # padtype="zero",
    return_H=False,
    precomped_H=None,
    return_H_exp=False,
    precomped_H_exp=None,
    dtype=torch.float32,
):
    device = u_in.device

    if return_H or return_H_exp:
        return angular_spectrum(
            u_in=u_in[0, 0, :, :],
            wv=wavelength,
            d1=feature_size[0],
            dz=z,
            # linear_conv=True,
            # padtype="zero",
            return_H=return_H,
            H=precomped_H,
            return_H_exp=return_H_exp,
            H_exp=precomped_H_exp,
            dtype=dtype,
            device=device,
        )

    res, _, _ = angular_spectrum(
        u_in=u_in[0, 0, :, :],
        wv=wavelength,
        d1=feature_size[0],
        dz=z,
        # linear_conv=True,
        # padtype="zero",
        return_H=False,
        H=precomped_H,
        return_H_exp=False,
        H_exp=precomped_H_exp,
        dtype=dtype,
        device=device,
    )

    return res[None, None, :, :]


def prop_waveprop_asm_lens(  # TODO does not work
    u_in,
    feature_size,
    wavelength,
    z,
    # linear_conv=True, # TODO check those two parameters
    # padtype="zero",
    return_H=False,
    precomped_H=None,
    return_H_exp=False,
    precomped_H_exp=None,
    dtype=torch.float32,
):
    res = prop_waveprop_asm(
        u_in,
        feature_size,
        wavelength,
        z,
        # linear_conv,
        # padtype,
        return_H,
        precomped_H,
        return_H_exp,
        precomped_H_exp,
        dtype,
    )[None, None, :, :]

    print(res.shape)
    print(u_in.shape[2:])
    print(feature_size)

    return neural_holography_lensless_to_lens(res, z, wavelength, u_in.shape[2:], feature_size)
