import imageio
import os
import skimage.io
import torch
import numpy as np

from slm_controller.hardware import (
    SLMParam,
    slm_devices,
)

from slm_designer.hardware import (
    CamParam,
    cam_devices,
)

from slm_designer.experimental_setup import (
    PhysicalParams,
    physical_params,
    slm_device,
    cam_device,
)

from slm_designer.wrapper import (
    ModelPropagate,
    propagation_ASM,
    ImageLoader,
    make_kernel_gaussian,
    crop_image,
    polar_to_rect,
    propagate_field,
    srgb_lin2gamma,
    cond_mkdir,
    PhysicalProp,
)


def citl_predict():  # TODO Use click
    channel = 1
    prop_model = "ASM"
    root_path = "./phases"
    prop_model_dir = "./calibrated_models"
    calibration_path = "./calibration"

    # Parse
    # opt = p.parse_args()
    # channel = channel
    chs = range(channel) if channel == 3 else [channel]  # retrieve all channels if channel is 3
    chan_strs = ("red", "green", "blue", "rgb")
    run_id = f'{root_path.split("/")[-1]}_{prop_model}'  # {algorithm}_{prop_model}

    prop_dist = physical_params[
        PhysicalParams.PROPAGATION_DISTANCE
    ]  # propagation distance from SLM plane to target plane
    wavelength = physical_params[PhysicalParams.WAVELENGTH]  # wavelength
    pixel_pitch = slm_devices[slm_device][SLMParam.PIXEL_PITCH]  # SLM pitch

    slm_res = slm_devices[slm_device][SLMParam.SLM_SHAPE]  # resolution of SLM
    image_res = cam_devices[cam_device][CamParam.IMG_SHAPE]  # TODO slm.shape == image.shape?
    roi_res = (round(slm_res[0] * 0.8), round(slm_res[1] * 0.8))

    # # Resolutions
    # slm_res = (1080, 1920)  # resolution of SLM
    # if "HOLONET" in run_id.upper():
    #     slm_res = (1072, 1920)
    # elif "UNET" in run_id.upper():
    #     slm_res = (1024, 2048)

    # image_res = (1080, 1920)
    # roi_res = (880, 1600)  # regions of interest (to penalize)
    dtype = (
        torch.float32
    )  # default datatype (Note: the result may be slightly different if you use float64, etc.)
    device = torch.device("cuda")  # The gpu you are using

    # You can pre-compute kernels for fast-computation
    precomputed_H = [None] * 3
    if prop_model == "ASM":
        propagator = propagation_ASM
        for c in chs:
            precomputed_H[c] = propagator(
                torch.empty(1, 1, *slm_res, 2), pixel_pitch, wavelength, prop_dist, return_H=True,
            ).to(device)

    elif prop_model.upper() == "CAMERA":
        propagator = PhysicalProp(
            channel,
            laser_arduino=True,
            roi_res=(roi_res[1], roi_res[0]),
            slm_settle_time=0.15,
            range_row=(220, 1000),
            range_col=(300, 1630),
            patterns_path=calibration_path,  # path of 21 x 12 calibration patterns, see Supplement.
            show_preview=True,
        )
    elif prop_model.upper() == "MODEL":
        blur = make_kernel_gaussian(0.85, 3)
        propagators = {}
        for c in chs:
            propagator = ModelPropagate(
                distance=prop_dist, feature_size=pixel_pitch, wavelength=wavelength, blur=blur,
            ).to(device)

            propagator.load_state_dict(
                torch.load(
                    os.path.join(prop_model_dir, f"{chan_strs[c]}.pth"), map_location=device,
                )
            )
            propagator.eval()
            propagators[c] = propagator

    print(f"  - reconstruction with {prop_model}... ")

    # Data path
    data_path = "./data/test"
    recon_path = "./recon"

    # Augmented image loader (if you want to shuffle, augment dataset, put options accordingly.)
    image_loader = ImageLoader(
        data_path,
        channel=channel if channel < 3 else None,
        image_res=image_res,
        homography_res=roi_res,
        crop_to_homography=True,
        shuffle=False,
        vertical_flips=False,
        horizontal_flips=False,
    )

    # Placeholders for metrics
    # psnrs = {"amp": [], "lin": [], "srgb": []}
    # ssims = {"amp": [], "lin": [], "srgb": []}
    idxs = []

    # Loop over the dataset
    for target in image_loader:
        # get target image
        target_amp, target_res, target_filename = target
        target_path, target_filename = os.path.split(target_filename[0])
        target_idx = target_filename.split("_")[-1]
        target_amp = target_amp.to(device)

        print(f"    - running for img_{target_idx}...")

        # crop to ROI
        target_amp = crop_image(target_amp, target_shape=roi_res, stacked_complex=False).to(device)

        recon_amp = []

        # for each channel, propagate wave from the SLM plane to the image plane and get the reconstructed image.
        for c in chs:
            # load and invert phase (our SLM setup)
            phase_filename = os.path.join(root_path, chan_strs[c], f"{target_idx}.png")
            slm_phase = skimage.io.imread(phase_filename) / 255.0
            slm_phase = (
                torch.tensor((1 - slm_phase) * 2 * np.pi - np.pi, dtype=dtype)
                .reshape(1, 1, *slm_res)
                .to(device)
            )

            # propagate field
            real, imag = polar_to_rect(torch.ones_like(slm_phase), slm_phase)
            slm_field = torch.complex(real, imag)

            if prop_model.upper() == "MODEL":
                propagator = propagators[c]  # Select CITL-calibrated models for each channel
            recon_field = propagate_field(
                slm_field, propagator, prop_dist, wavelength, pixel_pitch, prop_model, dtype,
            )

            # cartesian to polar coordinate
            recon_amp_c = recon_field.abs()

            # crop to ROI
            recon_amp_c = crop_image(recon_amp_c, target_shape=roi_res, stacked_complex=False)

            # append to list
            recon_amp.append(recon_amp_c)

        # list to tensor, scaling
        recon_amp = torch.cat(recon_amp, dim=1)
        recon_amp *= torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True) / torch.sum(
            recon_amp * recon_amp, (-2, -1), keepdim=True
        )

        # tensor to numpy
        recon_amp = recon_amp.squeeze().cpu().detach().numpy()
        target_amp = target_amp.squeeze().cpu().detach().numpy()

        if channel == 3:
            recon_amp = recon_amp.transpose(1, 2, 0)
            target_amp = target_amp.transpose(1, 2, 0)

        # calculate metrics
        # psnr_val, ssim_val = utils.get_psnr_ssim(
        #     recon_amp, target_amp, multichannel=(channel == 3)
        # )

        idxs.append(target_idx)

        # for domain in ["amp", "lin", "srgb"]:
        #     psnrs[domain].append(psnr_val[domain])
        #     ssims[domain].append(ssim_val[domain])
        #     print(
        #         f"PSNR({domain}): {psnr_val[domain]},  SSIM({domain}): {ssim_val[domain]:.4f}, "
        #     )

        # save reconstructed image in srgb domain
        recon_srgb = srgb_lin2gamma(np.clip(recon_amp ** 2, 0.0, 1.0))
        cond_mkdir(recon_path)
        imageio.imwrite(
            os.path.join(recon_path, f"{target_idx}_{run_id}_{chan_strs[channel]}.png"),
            (recon_srgb * np.iinfo(np.uint8).max).round().astype(np.uint8),
        )


if __name__ == "__main__":
    citl_predict()
