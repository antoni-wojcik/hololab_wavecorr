# HoloLab_wavecorr
A Python package for measuring the phase and amplitude of incident wavefronts in SLM-based Fourier optical systems to characterize and correct aberrations. It can measure wavefronts, compute and apply affine transforms between experimental and target images, and generate holograms incorporating aberration corrections. 

The package contains code used to obtain results for the *"Fourier-plane wavefront and SLM aberration characterization via iterative scanning of beam deflector segments"* paper.

This package is a subset of a larger codebase, containing only the functionality directly relevant to the methods and results presented in the associated paper.

## Author
This package has been created by Antoni J. Wojcik during his PhD at the University of Cambridge, CMMPE group, supervised by Prof. Timothy D. Wilkinson.

## Documentation
The package is divided into two sections: experiments and source code. 

The [experiments](experiments) directory contains scripts to:
1. measure the wavefront
2. generate the hologram based on a target, project, and capture the image
3. find the affine transform between the target and the obtained image

The [src](src) directory contains classes and functions used by the scripts used in experiments.

It additionally creates "[data](data)/experiments" directory when any experiments are performed in order to store the output data.

The package was programmed to work seamlesly with **VSCode** to resolve paths between the directories. If using alternative IDEs, the paths have to be set accordingly.

The package was programmed specifically to operate with Santec SLM-200 spatial light modulator (https://www.santec.com/en/products/components/slm/slm-200/), and Zelux 1.6 MP CMOS camera (https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=13677#ad-image-0). The libraries to operate these devices are not included and should be copied to the respective subdirectories in the [hardware](src/hardware) directory. If any other components are to be used, they have to be added in that directory.

The code uses packages to load SVG images as target patterns for hologram generation. It requires either the **CairoSVG** package or **Inkscape** program to be installed.