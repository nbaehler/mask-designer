mask-designer
=============

Collection of techniques to design mask patterns, e.g. fixed apertures for
amplitude masks and phase retrieval for determining a phase mask under an
incoherent light source.

.. contents:: Table of Contents
   :depth: 5
.. :local:
.. :backlinks: none



The main goal of the project is tackle the inverse problem called phase retrieval,
i.e. mask design for SLMs.
But it also allows to explore the forward problem by setting a phase
mask and then simply observing the output amplitude at the target plane.
Mainly though, it consists of mask design techniques that were introduced in the `Neural
Holography <https://www.computationalimaging.org/publications/neuralholography/>`_
work by Prof. Gordon Wetzstein's group at Stanford, and adapted to be compliant
with the SLM of this project - the `Holoeye LC
2012 <https://holoeye.com/lc-2012-spatial-light-modulator/>`_. Further, another
part of the code does add support for cameras. Neural Holography, amongst
others, follows a **Camera-In-The-Loop** approach which involves a
camera taking pictures of the resulting interference patterns at the target
plane and then using this information to improve the designed mask iteratively.
Finally, utility functions are provided so that masks designed by Holoeye's
(closed-source) software or Neural Holography's code can be used
interchangeably for either setup (as Holoeye and Neural Holography assume
different physical setups).

If you wish to learn more about the evolution of our experimental setup, please
refer to the ``docs/DOCUMENTATION.md``.


Below is a schematic of how ``mask-designer`` would typically interact with other
components.

.. image:: images/structure.svg
   :target: images/structure.svg
   :align: center
   :alt: Structure

The interactions marked with *CITL* are only necessary for the CITL approach.
You can find animated gif-files showing those interactions in more details in
``documentation/DOCUMENTATION.md``.

Installation
------------

To install, simply run the following script:

.. code-block:: sh

   ./env_setup.sh

The script will:


#. Create a Python3 virtual environment called ``mask_designer_env``.
#. Install Python dependencies in the virtual environment.
#. Install both `slm-controller <https://github.com/ebezzam/slm-controller>`_ and
   `waveprop <https://github.com/ebezzam/waveprop>`_ in setuptools “develop mode”
   from GitHub directly.

This project is using those two repositories to access physical SLMs after the
phase mask has been computed and to simulate the light propagation in the different
phase retrieval algorithms or propagation simulations. Note that those are still
in development too.

If you plan to use this code base more in depth you can install additional
dependencies intended for developing while the virtual environment is activated.

.. code-block:: sh

   source mask_designer_env/bin/activate
   pip install click black pytest tensorboard torch_tb_profiler sphinx-rtd-theme docutils==0.16


Optional setup for Camera-In-The-Loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Camera-In-The-Loop (CITL) obviously requires a camera. For this project we made
use of this camera `Thorlabs
DCC3260M <https://www.thorlabs.com/thorproduct.cfm?partnumber=DCC3260M>`_ (which is unfortunately obsolete). If you would like to use
the same camera for the CITL technique introduced by Neural Holography, you will
have to install IDS software as well. First, install `IDS Software
Suite <https://en.ids-imaging.com/download-details/AB00695.html>`_, simply follow
the installation instructions. Next, you need `IDS
Peak <https://en.ids-imaging.com/download-details/AB00695.html>`_ which can be
found under the same link.
This is a software package containing some GUI applications to interact with the
camera but, most importantly, contains two wheel-packages that allow to install
the python API for the aforementioned camera. First, just follow the install
instructions but then make sure to pick the ``Custom`` installation option to add an
important feature. Once in this selection prompt,
check the box additionally installing ``Support of uEye cameras`` and continue.
After the installation is completed go to the installation directory. Note
that you should have activated the virtual environment created earlier from now
on (``source mask_designer_env/bin/activate``). Next, go to ``ids_peak/generic_sdk/api/binding/python/wheel`` and
install the appropriate wheel like so, for example:

.. code-block:: sh

   pip install x86_64/ids_peak-1.4.1.0-cp39-cp39-win_amd64.whl

Secondly, again from the IDS installation
directory, go to ``ids_peak/generic_sdk/ipl/binding/python/wheel``. Similar to
before, install the correct version of the wheel for your setup, for example:

.. code-block:: sh

   pip install x86_64/ids_peak_ipl-1.3.2.7-cp39-cp39-win_amd64.whl

Now, you should be good to go to use all the features implemented in this
project.

Phase retrieval algorithms
--------------------------

The authors of `Neural Holography <https://www.computationalimaging.org/publications/neuralholography/>`_
provide implementations to different phase retrieval approaches. Here is a list
of methods that were modified in order to be compatible with the hardware and
software components as shown in the above schematic:


* Gerchberg-Saxton (GS)
* Stochastic Gradient Descent (SGD)
* Camera-In-The-Loop (CITL)

GS and SGD are implemented inside ``mask_designer/neural_holography/algorithms.py``
and PyTorch modules that go along with them are provided in
``mask_designer/neural_holography/module.py``. CITL on the other hand is located in a separate
script ``mask_designer/neural_holography/train_model.py``. Note that you do
generally not need to interact with the Neural Holography code directly. A
wrapper for it is provided at ``mask_designer/wrapper.py`` which does simply import
code from Neural Holography so that you do not need to go look for it in their
code and also contains some interfacing methods to run the different phase
retrieval algorithms. We'd like to remind that
this code was released under the license provided in ``LICENSE`` and we do not
claim any credit for it. Usage examples of all
those features will be presented in the
subsequent `Example scripts <#example-scripts>`_ section.


.. TODO add our own license



Camera
------

As mentioned earlier, cameras play a crucial role in the CITL-approach. Hence, an
interface for such devices is needed. For now, the project only supports one
real camera, the `Thorlabs
DCC3260M <https://www.thorlabs.com/thorproduct.cfm?partnumber=DCC3260M>`_ and a
dummy camera that simply "takes" black snapshots. The later can be useful during
development. In the future this list is going to be extended (for example with
the `Raspberry Pi HQ Camera <https://www.adafruit.com/product/4561>`_), but here
is its current state.

Supported cameras:


* Dummy camera (artificial, returns synthetic pitch black images)
* `Thorlabs DCC3260M <https://www.thorlabs.com/thorproduct.cfm?partnumber=DCC3260M>`_

Experimental setup
------------------

The experimental setup is an incremental improvement of an initial setup proposed
by Holoeye in the manual that came with their their `LC 2012
SLM <https://holoeye.com/lc-2012-spatial-light-modulator/>`_. For more information
on how we converged to the setup below, please refer to ``documentation/DOCUMENTATION.md``.


.. image:: images/setup.svg
   :target: images/setup.svg
   :align: center
   :alt: Experimental setup



Further, the ``mask_designer/experimental_setup.py`` allows one to set:


* which camera and SLM are used,
* how long masks are shown on the SLM,
* what wavelength the laser is operating at and, finally,
* the propagation distance (distance form the SLM to the camera sensor).

Those parameters are then used in the remainder of the code base.


.. TODO might not be only linked to lenses, ASM vs Fraunhofer



Holoeye provides a graphical software called `SLM Pattern
Generator <https://customers.holoeye.com/slm-pattern-generator-v5-1-1-windows/>`_
that has built-in functionality for performing phase retrieval for a given
target amplitude. One such example can be found in ``images/phase_mask``
and its corresponding amplitude at the target plane under ``images/target_amplitude``.
This software assumes an experimental setup that uses a convex lens in between the SLM and
the target plane.

Neural Holography on the other hand, uses a different setting
where no lens is placed between the SLM and the target plane, i.e. a lensless
setting. Those differences impact the resulting phase masks of the mask design
algorithm. The methods in ``mask_designer/transform_fields.py`` allow
transforming phase maps, or fields, back and forth between both experimental
setups. Note that Neural Holography encodes
phase maps, images etc. as 4D PyTorch Tensors where the dimensions are [image,
channel, height, width]. But again, the wrapper ``mask_designer/wrapper.py`` does
provide interfacing methods for the different algorithms that handle all those
complications for you and you are not required to dig any deeper than that.

Propagation
-----------


.. TODO might not be only linked to lenses, ASM vs Fraunhofer



This section will briefly discuss the propagation of a phase mask to the target
plane. More precisely, propagation simulation is a crucial element in most of the
mask designing algorithms. Although we cannot be absolutely certain due to the code being closed-source, we
have good reason to believe that Holoeye's SLM Pattern Generator uses
`Fraunhofer <https://en.wikipedia.org/wiki/Fraunhofer_diffraction_equation>`_, as
we have identified a single Fourier Transform between the SLM and target plane
when playing around with their masks. Neural Holography on the other hand,
uses the `Angular spectrum
method <https://en.wikipedia.org/wiki/Angular_spectrum_method>`_ (ASM). Currently,
we make use of the ASM implementation by Neural Holography. However we plan to
replace this implementation with the
`waveprop <https://github.com/ebezzam/waveprop>`_ library, which provides
support for Fraunhofer, ASM, and other propagation techniques.


.. TODO replace prop with waveprop



Physical propagation
^^^^^^^^^^^^^^^^^^^^

Physical propagation refers to the process of physically displaying a phase map
on a SLM and then observing the resulting images at the target plane. That's where the
`slm-controller <https://github.com/ebezzam/slm-controller>`_ comes in handy to
communicate with the physical SLMs, and the camera in order to measure the
response at the target plane.


.. TODO must change if we decide to always plot the masks



Note that this software package simply plots
the phase mask whenever something goes wrong with showing it on the physical
device so that you can still get an idea of the resulting phase maps.

Usage examples will be presented in the
subsequent `Example scripts <#example-scripts>`_ section.

Setting common apertures
------------------------

The ``mask_designer/aperture.py`` provides
an easy way to set different apertures: rectangle, square, line, and circle.
These apertures can be programmed to real SLM devices. Usage example will be
presented in the subsequent `Example scripts <#example-scripts>`_ section.


Adding a new camera
-------------------

In order to add support for a new camera, a few steps need to be taken. These are
done to avoid hard-coded values, but rather have global variables/definitions
that are accessible throughout the whole code base.


#. Add camera configuration in ``mask_designer/hardware.py:cam_devices``.
#. Define a new class in ``mask_designer/camera.py`` for interfacing with the new
   camera (set parameters, take images, etc.).
#. Add to factory method ``create`` in ``mask_designer/camera.py`` for a
   conveniently one-liner to instantiate an object of the new camera.

Issues
------

Currently, we aren't aware of any issues. If you should find any, please let us know.

Future work
-----------

Here, we list features and directions we want to explore in future work.

1. Check if Windows can be run in a container on a Raspberry Pi.
