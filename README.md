# BINDER
Bayesian INference for DEformable Registration

## Build Status

| Linux   | Windows    | MacOS |
|---------|------------|-----|
| ![Build Status](https://github.com/ste93ste/BINDER_private/actions/workflows/linux.yml/badge.svg) | ![Build Status](https://github.com/ste93ste/BINDER_private/actions/workflows/windows.yml/badge.svg) | ![Build Status](https://github.com/ste93ste/BINDER_private/actions/workflows/macos.yml/badge.svg) |

## Installation

Install the Python package with:

 `pip install .` 

By default, the installation will attempt to install the GPU version (if CUDA is available). 
If you want to force the CPU installation you can do it by:

 `pip install . -Ccmake.define.BINDER_FORCE_CPU=ON` 

The topology correction algorithm is included by default. If you don't want it (see the
licensing note below), exclude it from the build with:

 `pip install . -Ccmake.define.BINDER_DISABLE_TOPOLOGY_CORRECTION=ON`

> **Note on licensing:** The topology correction algorithm is derived from the FSL's implementation of FNIRT,
> which is distributed under the [FSL license](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html) —
> free for non-commercial use only. It is included by default; build with
> `-Ccmake.define.BINDER_DISABLE_TOPOLOGY_CORRECTION=ON` to exclude it.

### CPU backend dependencies

The CPU backend requires the [FFTW](https://www.fftw.org/) and [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) libraries. These are **not** needed for the GPU build. On Linux you can install both with:

 `sudo apt install libeigen3-dev libfftw3-dev`

> **Note on licensing:** The CPU backend links against [FFTW](https://www.fftw.org/), which is
> distributed under the GPL. Building the CPU version therefore makes the resulting binary subject
> to the GPL.

## Scripts
The main entry point is `run_BINDER`. An example call:

`run_BINDER --moving MovingImage.nii.gz --fixed FixedImage.nii.gz --output outputDirectory --save-warped-image`

`applyField.py` warps an image using an already-estimated deformation field.

## Examples
The `examples` folder contains ready-to-run scripts that point at images already in this repository:

- [2D_MI_registration.py](BINDER/examples/2D_MI_registration.py) — 2D registration with the MI metric
- [3D_MI_registration.py](BINDER/examples/3D_MI_registration.py) — 3D registration with the MI metric

## Citation

If you use BINDER in your research, please cite:

```bibtex
@article{cerri_binder,
  title   = {{BINDER}: A Latent Variable Model for Probabilistic Medical Image Registration},
  author  = {Cerri, Stefano and Hassankhani, Amirhossein and Balbastre, Yael and Van Leemput, Koen},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
  }
```