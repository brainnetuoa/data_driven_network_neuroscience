[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Data-Driven Network Neuroscience: On Data Collection and Benchmark

This repository contains a package of scripts and codes used in the paper to convert raw functional images to connectivity matrices using [fMRIPrep](https://fmriprep.org/en/stable/). The detailed description can be found in [our paper](https://arxiv.org/abs/2211.12421) on NeurIPS, Datasets and Benchmarks Track, 2023. The dataset we used and the preprocessed data is collected at [https://doi.org/10.17608/k6.auckland.21397377](https://doi.org/10.17608/k6.auckland.21397377).

## Samples of the raw MRI / preprocessed outputs / matrices  

* A sample TaoWu subject can be accessed here: `to be updated`
* The fMRIPrep outputs of the same subject fully preprocessed can be accessed here: `to be updated`
* The set of connectivity matrices (using correlation) generated for the same subject can be accessed here: `to be updated`

## Requirements
* Containerized execution environment: 
	* [Docker](https://www.nipreps.org/apps/docker/) or 
	* [Singularity](https://www.nipreps.org/apps/singularity/)

## External Dependencies
* [fMRIPrep](https://fmriprep.org/en/stable/index.html) (version 20.2.3)
* [dcm2niix](https://github.com/rordenlab/dcm2niix)
* [nilearn](https://nilearn.github.io/stable/index.html)

## Setup

1. Install numpy, os, shutil, glob, dcm2niix, nilearn, scipy modules for python programming
2. Install Docker/Singularity and fMRIPrep 
	* Installing fMRIPrep requires several steps and sorting out dependencies and a freesurfer license (free to acquire). We recommend following [this guide] (https://andysbrainbook.readthedocs.io/en/latest/OpenScience/OS/fMRIPrep.html)

## Steps to preprocess neuroimages:

![alt text](https://raw.githubusercontent.com/bna-data-analysis/extract-brain-network/main/asset/nips_flowchart.png)

## Step A: Data Collection and Selection
Access Link for [Neurocon and TaoWu Dataset](http://fcon_1000.projects.nitrc.org/indi/retro/parkinsons.html).

Access Link for [ABIDE/ADNI/PPMI](https://ida.loni.usc.edu/login.jsp). 

* A sample TaoWu subject can be accessed here: `to be updated`
* Each fMRI image needs to be accompanied with a structural T1-weighted (T1w) image acquired from the same subject (ideally) in the same scan session

## Step B: BIDS Format Conversion

Raw T1w/fMRI data are in DICOM or NifTi format 

* This step is to convert raw MRI data in either DICOM or NifTi into [BIDS format](https://bids.neuroimaging.io/)
	* ABIDE_Nifti2BIDS.py - To convert raw MRI data (NifTi format) to BIDS - For ABIDE dataset
	* ADNI_PPMI_DCM2BIDS.py - To convert raw MRI data (DICOM format) to BIDS - For PPMI and ADNI dataset
	* Neurocon and TaoWu dataset are NifTi files and are already BIDS formatted

## Step C: fMRIPrep Preprocessing

Make sure you have installed fMRIPrep correctly using the information and guides from the links above.

* Preprocess BIDS formatted neuroimages (1 T1w image and 1 fMRI BOLD image) using fMRIPrep
	* fmriprep_shellscript.sh - Script to execute fmriprep preprocessing
* fMRIPrep outputs a number of BIDS Derivative compliant files
	* For details on each file: https://fmriprep.org/en/stable/outputs.html

A sample of a fully preprocessed TaoWu subject (and its outputs) can be accessed here: `to be updated`

## Steps D, E, and F: Parcellation and ROI Definition, Connectivity Matrix Extraction, and Graphical Brain Network

* Convert preprocessed Nifti images into connectivity matrices
	* ConnectivityMatrices.py - Code to generate connectivity matrices

## Perform Experimental Analysis

* The codes we used to run our empirical analysis 
	* NIPS_paper_gridsearch_experiments.py
	* EdgeWeightsStatistics.py

Note: Input/Output location and the required modification are detailed within the python codes

## Reference

```
@misc{xu2023datadriven,
      title={Data-Driven Network Neuroscience: On Data Collection and Benchmark}, 
      author={Jiaxing Xu and Yunhan Yang and David Tse Jung Huang and Sophi Shilpa Gururajapathy and Yiping Ke and Miao Qiao and Alan Wang and Haribalan Kumar and Josh McGeown and Eryn Kwon},
      year={2023},
      eprint={2211.12421},
      archivePrefix={arXiv},
      primaryClass={q-bio.NC}
}
```