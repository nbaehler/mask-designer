# Documentation

- [Documentation](#documentation)
  - [Overview](#overview)
  - [Experimental setup](#experimental-setup)
  - [Propagation](#propagation)

## Overview

![Schematic representation of the interactions between different components](images/structure.svg)
![Holoeye logo](../images/target_amplitude/holoeye_logo.png)
![Holoeye logo phase](../images/holoeye_phase_map/holoeye_logo_slm_pattern.png)
![Holoeye logo propagated](images/holoeye_logo_propagated.png)

## Experimental setup

![Experimental setup](images/setup_0.svg)
![Experimental setup](images/setup_1.svg)
![Experimental setup](images/setup_2.svg)
![Experimental setup](images/setup_3.svg)
![Experimental setup](images/setup_4.svg)

![Experimental setup](images/setup.svg)
![Experimental setup](images/setup.jpg)

![Lenses diagram](images/lenses_diagram.svg)

$$
\begin{align}
 \frac{1}{−75}=\frac{1}{b}−\frac{1}{a} &\iff \frac{1}{b}=\frac{1}{a}−\frac{1}{75} \\
 &\iff b=\left(\frac{1}{a}−\frac{1}{75}\right)^{−1} \\
 &\iff b=\left(\frac{1}{200−c}−\frac{1}{75}\right)^{−1}
\end{align}
$$

with $a=200−c$ and $125 < c < 200$.

## Propagation

![Neural Holography experimental setup](images/neural_holography_setup.png)
![Experimental setup](images/setup.svg)
![Different propagation methods used by Holoeye and Neural Holography](images/propagation.svg)

$$
\begin{align}
A_h&=(FT \circ S)(\phi_h) \\
A_n&=(IS \circ FT \circ S \circ M \circ IFT \circ S)(\phi_n) \\
\phi_n&=(IS \circ FT \circ S \circ M \circ IFT \circ IFT)(\phi_h) \\
\phi_h&=(FT \circ FT \circ S \circ M^{-1} \circ IFT \circ S)(\phi_n) \\
\end{align}
$$

where $FT$ is a regular Fourier transform, $IFT$ its inverse transform,
$S$ simply shifts i.e. rotates part of the Tensors, $IS$ does the inverse shift
and $M$ is a matrix multiplication by the homography matrix $H$ computed internally.
