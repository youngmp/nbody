
---
title: "Generating Higher-Order Coupling Functions for N-Body Oscillator Interations: A Python Library"
author:
- name: Youngmin Park
  affiliation: University of Florida
- name: Dan D. Wilson
  affiliation: University of Tennessee Knoxville
output:
  pdf_document
abstract: "We document the nBodyCoupling library. The framework is reasonably general, with no a priori restrictions on model dimension or type of coupling function."

...



# Introduction

nBodyCoupling is a script for computing the higher-order coupling functions in my paper with Dan Wilson, ``High-Order Accuracy Compuation of Coupling Functions for Strongly Coupled Oscillators''. The script generates higher-order interaction functions for phase reductions of systems containing limit cycle oscillations.

## Dependencies

All following libraries are required to make the script run.

| Package	| Version	| Link		| 
| -----------	| -----------	| -----------	|
| Python	| 3.8.1		|
| Matplotlib	| 3.7.2		|		|
| Numpy		| 1.21.6	|		|
| Scipy		| 1.10.1	|		|
| Pathos	| 0.2.8		| https://anaconda.org/conda-forge/pathos |
| tqdm		| 4.30.0	| https://anaconda.org/conda-forge/tqdm |
| Sympy		| 1.12		| https://anaconda.org/anaconda/sympy |

Notes on depedendencies:

**Python 3.7+ is necessary**. Our code often requires an arbitrary number of function inputs, which earlier versions of Python do not allow. The script will likely work with earlier versions of all other libraries.

### Other Notes

Make sure to use **pathos** over multiprocessing because pickling is more robust with pathos. Pathos uses dill, which can serialize far more objects compared to multiprocessing, which uses pickle.

The code is written so that tqdm is necessary, but tqdm only provides a status bar during parallel computing. It is not part of the engine, and the code can be modified to work without it. In future versions I may leave tqdm as a toggle.

## Installation

As long as your computer has the packages listed above and they are installed using Python 3.7, the nBodyCoupling script should run. Just place it within the same working directory as your Python script and import it as a module.

I have no immediate plans to release the nBodyCoupling script as an installable package simply because I do not have the time to maintain and track version releases for distribution platforms such as anaconda, pip, and apt.

# Reproduce Figures

To reproduce the figures in Park and Wilson 2020, cd to the examples directory and run

   $ generate_figures.py

This file will call the complex Ginzburg-Landau (CGL) model file (CGL.py) and the thalamic model file (Thalamic.py) and generate the figures from the paper. I've taken care of most of the work and made sure that the code uses saved data files for figure generation as opposed to computing everything from scratch. If you don't have the data files, it will take a while to run, and will use 8 cores by default. Make sure to edit the keyword arguments (documented in the nBodyCoupling section below) if you wish to use more or less cores.
