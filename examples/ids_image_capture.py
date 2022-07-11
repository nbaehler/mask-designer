"""
IDS camera single image capture example.
"""

from slm_designer.hardware import CamDevices
from slm_designer import camera
import matplotlib.pyplot as plt


def main():
    # Initialize ids camera
    cam = camera.create_camera(CamDevices.IDS.value)

    # Acquire one image
    image = cam.acquire_images()

    # and plot it using matplotlib
    _, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
