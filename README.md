# Microbundle Compute Repository

<!---
We will configure these once we make the repository public:
[![python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/sandialabs/sibl#license)

[![tests](https://github.com/elejeune11/microbundlecompute/workflows/tests/badge.svg)](https://github.com/elejeune11/microbundlecompute/actions) [![codecov](https://codecov.io/gh/elejeune11/microbundlecompute/branch/main/graph/badge.svg?token=EVCCPWCUE7)](https://codecov.io/gh/elejeune11/microbundlecompute)
-->

## Table of Contents
* [Project Summary](#summary)
* [Project Roadmap](#roadmap)
* [Installation Instructions](#install)
* [Tutorial](#tutorial)
* [Validation](#validation)
* [To-Do List](#todo)
* [References to Related Work](#references)
* [Contact Information](#contact)
* [Acknowledgements](#acknowledge)

## Project Summary <a name="summary"></a>
The MicroBundleCompute software is developed as a multi-purpose tool for analyzing heterogeneous cardiac microtissue deformation and strain from brightfield movies of beating microtissue. In this repository, we share the source code, steps to download and install the software, tutorials on how to run the different main and optional functionalities of the software, and details about code validation. **For more information, please refer to the main [manuscript](add link).**

Briefly, the software requires two main inputs: `1)` a binary mask of the tissue and `2)` consecutive movie frames of the beating microtissue. The mask can be either generated manually or externally, or automatically using one of the software’s built-in functionalities. Tracking points identified as Shi-Tomasi corner points are then computed on the first frame of the movie and tracked
across all frames. From this preliminary tracking, we can identify individual beats. This allows us to perform the analysis per beat by tracking the
marker points identified at the first frame of each beat across the beat frames. From these tracked points, we are able to compute full-field displacements,
and subdomain-averaged strains. We also include post-processing functionalities to rotate the images and tracking results as well as interpolate
the results at query points. To visualize the results, the software outputs timeseries plots per beat and movies of full-field results. Finally, we validate our software against synthetically generated beating microtissue data with a known ground truth.

<p align = "center">
<img alt="code pipeline" src="tutorials/figs/code_pipeline.png" width="95%" />

Additionally, the user can also specify to track the pillars or posts to which the microtissue is attached. In this case, a mask for the pillars (posts) should be provided. The outputs for this tracking option are timeseries plots of the pillars' mean absolute displacement and force results. We note that this additional functionality has not been vigorously validated at the moment.

We are also adding new functionalities to the code as well as enhancing the software based on user feedback. Please check our [to-do list]((#todo)).

## Project Roadmap <a name="roadmap"></a>
The ultimate goal of this project is to develop and disseminate a comprehensive software for data curation and analysis from lab-grown cardiac microtissue on different experimental constructs. Prior to the initial dissemination of the current version, we have tested our code on approximately 30 examples provided by 3 different labs who implement different techniques. This allowed us to identify challenging examples for the software and improve our approach. We hope to further expand both our testing dataset and list of contributors.
The roadmap for this collaborative endeavor is as follows:

`Preliminary Dataset + Software` $\mapsto$ `Published Software Package` $\mapsto$ `Published Validation Examples and Tutorial` $\mapsto$ `Larger Dataset + Software Testing and Validation` $\mapsto$ `Automated Analysis of High-Throughput Experiments`

At present (**march 2023**), we have validated our software on a preliminary dataset in addition to a synthetically generated dataset (please find more details on the [SyntheticMicroBundle github page](https://github.com/HibaKob/SyntheticMicroBundle) and the [main manuscript](add link)). We also include details on validation against manual tracking [here](add link to SA). In the next stage, we are particularly interested in expanding our dataset and perform further software validation and testing. 
 Specifically, we aim to `1)` identify scenarios where our approach fails, `2)` create functions to accomodate these cases, and `3)` compare software results to previous manual approaches for extracting quantitative information, especially for pillar tracking. We will continue to update this repository as the project progresses.

## Installation Instructions <a name="install"></a>

### Get a copy of the microbundle compute repository on your local machine

The best way to do this is to create a GitHub account and ``clone`` the repository. However, you can also download the repository by clicking the green ``Code`` button and selecting ``Download ZIP``. Downloaded and unzip the ``microbundlecompute-main`` folder and place it in a convenient location on your computer.


### Create and activate a conda virtual environment

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/) on your local machine.
2. Open a ``Terminal`` session (or equivalent) -- note that Mac computers come with ``Terminal`` pre-installed (type ``⌘-space`` and then search for ``Terminal``).
3. Type in the terminal to create a virtual environment with conda:
```bash
conda create --name microbundle-compute-env python=3.9.13
```
4. Type in the terminal to activate your virtual environment:
```bash
conda activate microbundle-compute-env
```
5. Check to make sure that the correct version of python is running (should be ``3.9.13``)
```bash
python --version
```
6. Update some base modules (just in case)
```bash
pip install --upgrade pip setuptools wheel
```

Note that once you have created this virtual environment you can ``activate`` and ``deactivate`` it in the future -- it is not necessary to create a new virtual environment each time you want to run this code, you can simply type ``conda activate microbundle-compute-env`` and then pick up where you left off (see also: [conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)).


### Install microbundle compute

1. Use a ``Terminal`` session to navigate to the ``microbundlecompute-main`` folder. The command ``cd`` will allow you to do this (see: [terminal cheat sheet](https://terminalcheatsheet.com/))
2. Type the command ``ls`` and make sure that the file ``pyproject.toml`` is in the current directory.
3. Now, create an editable install of microbundle compute:
```bash
pip install -e .
```
4. If you would like to see what packages this has installed, you can type ``pip list``
5. You can test that the code is working with pytest (all tests should pass):
```bash
pytest -v --cov=microbundlecompute  --cov-report term-missing
```
6. To run the code from the terminal, simply start python (type ``python``) and then type ``from microbundlecompute import image_analysis as ia``. For example:
```bash
(microbundle-compute-env) eml-macbook-pro:microbundlecompute-main emma$ python
Python 3.9.13 | packaged by conda-forge | (main, May 27 2022, 17:01:00) 
[Clang 13.0.1 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> from microbundlecompute import image_analysis as ia
>>> ia.hello_microbundle_compute()
>>> "Hello World!
```

## Tutorial <a name="tutorial"></a>


## Validation <a name="validation"></a>


## To-Do List <a name="todo"></a>
- [ ] Adjust code so that the dictionaries for timeseries segmentation are computed automatically
- [ ] Adjust code so that the dictionaries for feature and tracking parameters can be conveniently adjusted
- [ ] Orient output (strain, displacement) direction with major axes of microbundle 
- [ ] Add functionality to compute beating frequency (might be useful to other groups)
- [ ] Ensure that division into subdomains does not result in singular matrices
- [ ] Validate displacement and strain output (synthetic data + manual tracking)
- [ ] Clean code
- [ ] Explore options for additional analysis/visualization
- [ ] Automate mask creation

## References to Related Work <a name="references"></a>


## Contact Information <a name="contact"></a>


## Acknowledgements <a name="acknowledge"></a>


