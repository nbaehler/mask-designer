"""
Camera image capture example.
"""

import sys
from os.path import abspath, dirname, join

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, ".."))
sys.path.append(CODE_DIR)

import matplotlib.pyplot as plt
from mask_designer import camera
from mask_designer.experimental_setup import cam_device
from mask_designer.simulate_prop import plot_mask


def main():
    # Initialize camera
    cam = camera.create(cam_device)

    # Acquire one image
    image = cam.acquire_single_image()

    # and plot it using matplotlib
    plot_mask(image)

    # Change the exposure time, take another image
    cam.set_exposure_time(35000)
    image = cam.acquire_single_image()

    # and plot it using matplotlib
    plot_mask(image)

    # Reset the exposure time to the default value, take 4 images
    cam.set_exposure_time()
    images = cam.acquire_multiple_images(4)

    # and plot them using matplotlib
    _, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(images[0], cmap="gray")
    ax[0, 1].imshow(images[1], cmap="gray")
    ax[1, 0].imshow(images[2], cmap="gray")
    ax[1, 1].imshow(images[3], cmap="gray")
    plt.show()

    # Acquire one image
    image = cam.acquire_single_image_and_resize_to_slm_shape()

    # and plot it using matplotlib
    plot_mask(image)

    # Change the exposure time, take another image
    cam.set_exposure_time(35000)
    image = cam.acquire_single_image_and_resize_to_slm_shape()

    # and plot it using matplotlib
    plot_mask(image)

    # Reset the exposure time to the default value, take 4 images
    cam.set_exposure_time()
    images = cam.acquire_multiple_images_and_resize_to_slm_shape(4)

    # and plot them using matplotlib
    _, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(images[0], cmap="gray")
    ax[0, 1].imshow(images[1], cmap="gray")
    ax[1, 0].imshow(images[2], cmap="gray")
    ax[1, 1].imshow(images[3], cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
