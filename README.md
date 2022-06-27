# slm-designer

Collection of different approaches to phase retrieval,
i.e. mask design.

## Overview

The main goal of the project is tackle the inverse problem of phase retrieval,
i.e. mask design.
But it also allows to explore the forward problem by setting the a phase
pattern and then observing the output amplitude at the target plane. Mainly though, it
consists of techniques that were introduced in [Neural
Holography](https://github.com/computational-imaging/neural-holography) and
and adapted to be compliant with the remainder of this project.
Further, another part of the code does add support for cameras. Neural
Holography amongst others follows a `Camera-In-The-Loop` approach which involves a
camera taking pictures of the resulting interference patterns at the target
plane and then using this information to improve the designed mask iteratively.
Finally, mostly for debugging purposes, functions that allow to transform
generated phase maps using Holoeye's software or the Neural Holography code into
each others hardware settings and simulated propagation functions for both
settings as well are provided.

## Installation

<!-- TODO Talk about dependencies with slm-controller and waveprop -->

```sh
./env_setup.sh
```

The script will:

1. Install OS dependencies. <!-- TODO Any? -->
2. Create a Python3 virtual environment called `slm_designer_env`.
3. Install Python dependencies in the virtual environment.

## Manual installation needed for enabling some features

If you plan to use the [Thorlabs
DCC3260M](https://www.thorlabs.com/thorproduct.cfm?partnumber=DCC3260M) camera
for the Camera-In-The-Loop phase retrieval method introduced by [Neural
Holography](https://github.com/computational-imaging/neural-holography) you will
have to install IDS software as well. First install [IDS Software
Suite](https://en.ids-imaging.com/download-details/AB00695.html), simply follow
the installation instructions. Next, you need [IDS
Peak](https://en.ids-imaging.com/download-details/AB00695.html) which can be
found at the same link.
This is a software package containing some GUI applications to interact with the
camera but most importantly contains two wheel-packages that allow to install
the API for the aforementioned camera. First, just follow the install
instructions and but then make sure to pick the `Custom` option to add an
important feature. Once in this selection prompt,
check the box additionally installing `Support of uEye cameras` and continue.
After the installation is completed go to the installation directory. Note
that you should have activated your virtual environment created earlier from now
on. Next, go to`ids_peak/generic_sdk/api/binding/python/wheel` and
install the appropriate wheel like so, for example:

```sh
pip install x86_64/ids_peak-1.4.1.0-cp39-cp39-win_amd64.whl
```

Secondly, a second needs to be installed. Again from the IDS installation
directory go to `ids_peak/generic_sdk/api/binding/python/wheel`. Similar to
before install the correct version of the wheel for your setup, for example:

```sh
pip install x86_64/ids_peak_ipl-1.3.2.7-cp39-cp39-win_amd64.whl
```

Now, you should be good to go to use all the features implemented in this
project.

## Neural Holography

The authors of the
[paper](https://www.computationalimaging.org/wp-content/uploads/2020/08/NeuralHolography_SIGAsia2020.pdf)
provide implementations to different phase retrieval approaches in their
[repository](https://github.com/computational-imaging/neural-holography). Here a
list of the slightly modified methods that are compatible
with the remainder of the project and where to find them:

1. Gerchberg-Saxton (GS)
2. Stochastic Gradient Descent (SGD)
3. Double Phase Amplitude Coding (DPAC)
4. Camera-In-The-Loop (CITL)

GS, SGD and DPAC are all implemented inside `slm_designer/neural_holography/algorithms.py`
and PyTorch modules that go along with them are provided in
`slm_designer/neural_holography/module.py`. CITL on the other hand is located in a separate
script `slm_designer/neural_holography/train_model.py`. This script is
currently not fully compatible with our setup and hence does crash for now when
computing the loss. This will be fixed in a next step. Usage examples of all
those features will be presented in the
subsequent [Example scripts](#example-scripts) section. Note that for now those methods are only
intended to be used with the `Holoeye LC 2012` and the `Thorlabs DCC3260M` camera,
hence some parameters are hardcoded to use those devices. In a later stage of the
project those sections will be made more modular.

## Propagation

This section will briefly discuss the propagation of a phase map to the target
plane. Holoeye does also provide a piece of software called [SLM Pattern
Generator](https://customers.holoeye.com/slm-pattern-generator-v5-1-1-windows/)
which amongst others has a feature that does perform phase retrieval for a given
target amplitude. One such example can be found in `images/holoeye_phase_map`.
This software assumes a hardware setting that uses a lens in between the SLM and
the target plane. Neural Holography on the other hand, uses a different setting
where no lens is placed between the SLM and the target plane, i.e. a lensless
setting. Those differences impact the resulting phase map of the mask design
algorithm. The methods in `slm_designer/transform_fields.py` allow transforming phase maps,
fields, back and forth between both settings. Note that Neural Holography encodes
phase maps, images etc. as 4D PyTorch Tensors where the dimensions are [image,
channel, height, width].

### Physical propagation

Physical propagation refers to the process of physically displaying a phase map
on a SLM and then observing the resulting images at the target plane. Usage
examples will be presented in the
subsequent [Example scripts](#example-scripts) section.

### Simulated propagation

Opposed to physical propagation, here the propagation is only simulated. No
physical SLM is involved. This feature is specially useful for working, testing,
debugging when not having access to all the material needed to do the physical
propagation. As explained earlier ([Propagation](#propagation)) Holoeye and
Neural Holography assume different hardware configuration. Hence, methods are
provided to simulated propagation in both settings in
`slm_designer/simulate_prop.py`. Usage examples will be presented in the
subsequent [Example scripts](#example-scripts) section.

## Example scripts

In `examples` are various example scripts that showcase all
the features integrated into the project.

First, activate the virtual environment:

```sh
source slm_designer_env/bin/activate
```

You can exit the virtual environment by running `deactivate`.

### Aperture

To set a defined aperture shape, check out the following script:

```sh
>> python examples/set_aperture.py --help

Usage: set_aperture.py [OPTIONS]

  Set aperture on a physical device.

Options:
  --shape [rect|square|line|circ]
                                  Shape of aperture.
  --n_cells INTEGER               Side length for 'square', length for 'line',
                                  radius for 'circ'. To set shape for 'rect',
                                  use`rect_shape`.

  --rect_shape INTEGER...         Shape for 'rect' in number of cells; `shape`
                                  must be set to 'rect'.

  --vertical                      Whether line should be vertical (True) or
                                  horizontal (False).

  --device [rgb|binary]           Which device to program with aperture.
  --help                          Show this message and exit.
```

For example, to create a circle aperture on the monochrome device with a radius of 20 cells:

```sh
python examples/set_aperture.py --device binary --shape circ --n_cells 20
```

For a square aperture on the RGB device with a side length of 2 cells:

```sh
python examples/set_aperture.py --device rgb --shape square --n_cells 2
```

You can preview an aperture with the following script. Note that it should be run on a machine with
plotting capabilities, i.e. with `matplotlib`.

```sh
>> python examples/plot_aperture.py --help

Usage: plot_aperture.py [OPTIONS]

  Plot SLM aperture.

Options:
  --shape [rect|square|line|circ]
                                  Shape of aperture.
  --n_cells INTEGER               Side length for 'square', length for 'line',
                                  radius for 'circ'. To set shape for 'rect',
                                  use`rect_shape`.

  --rect_shape INTEGER...         Shape for 'rect' in number of cells; `shape`
                                  must be set to 'rect'.

  --vertical                      Whether line should be vertical (True) or
                                  horizontal (False).

  --show_tick_labels              Whether or not to show cell values along
                                  axes.

  --pixel_pitch FLOAT...             Shape of cell in meters (height, width).
  --slm_shape INTEGER...          Dimension of SLM in number of cells (height,
                                  width).

  --monochrome                    Whether SLM is monochrome.
  --device [rgb|binary]           Which device to program with aperture.
  --help                          Show this message and exit.
```

For example, to plot a square aperture on the RGB device with a side length of 2 cells:

```sh
python examples/plot_aperture.py --shape square --n_cells 2 --device rgb
```

### Image acquisition using IDS software and Thorlabs camera

This file illustrates how a camera, here the `IDSCamera`, is instantiated and
use to take a single image.

```sh
python examples/ids_image_capture.py
```

### Physical propagation example

This section contains two scripts, one for sending a phase map created using the
Holoeye software to the Holoeye SLM and another one doing the same for sending
phase maps generated using Neural Holography methods. Note that the
transformation method for the phase maps implemented in
`slm_designer/transform_fields.py` are crucial here.

```sh
>> python examples/physical_prop_holoeye.py --help
Usage: physical_prop_holoeye.py [OPTIONS]

Options:
  --show_time FLOAT  Time to show the pattern on the SLM.
  --help             Show this message and exit.
```

```sh
>> python examples/physical_prop_neural_holography.py --help
Usage: physical_prop_neural_holography.py [OPTIONS]

Options:
  --show_time FLOAT  Time to show the pattern on the SLM.
  --help             Show this message and exit.
```

### Simulated propagation example

Same as above, two versions of the simulated propagation do exist, one for
Holoeye phase maps and another one for the phase maps computed with Neural
Holography methods. Again, the functions in `slm_designer/transform_fields.py`
are important here. As a sanity check each phase map is transformed into both
settings and the its propagation is simulated in the respective setting. The
resulting amplitude patterns must be the same.

```sh
python examples/simulate_prop_holoeye.py
```

```sh
python examples/simulate_prop_neural_holography.py
```

## Adding a new camera

1. Add configuration in `slm_designer/hardware.py:cam_devices`.
2. Create class in `slm_designer/camera.py`.
3. Add to factory method `create_camera` in `slm_designer/camera.py`.
