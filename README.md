# hololab_wavecorr
A Python package for measuring the phase and amplitude of incident wavefronts in SLM-based Fourier optical systems to characterize and correct aberrations. It can measure wavefronts, compute and apply affine transforms between experimental and target images, and generate holograms incorporating aberration corrections.

The package contains code used to obtain results for the *"Fourier-plane wavefront and SLM aberrationcharacterization via iterative scanning of beamdeflector segments"* paper.

## Author
This package has been created by Antoni Jan Wojcik during his PhD at the University of Cambridge, CMMPE group, supervised by Prof. Timothy D. Wilkinson.

## Documentation
The package is divided into two sections: experiments and classes. 

The [experiments](experiments) directory contains scripts to:
1. measure the wavefront
2. generate the hologram based on a target, project, and capture the image
3. find the affine trasnform between the target and the obtained image

The [src](src) directory contains classes and functions used by the scripts used in experiments.
