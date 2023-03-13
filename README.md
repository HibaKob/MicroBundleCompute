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

At present (**march 2023**), we have validated our software on a preliminary dataset in addition to a synthetically generated dataset (please find more details on the [SyntheticMicroBundle github page](https://github.com/HibaKob/SyntheticMicroBundle) and the [main manuscript](**add link**)). We also include details on validation against manual tracking [here](**add link to SA**). In the next stage, we are particularly interested in expanding our dataset and perform further software validation and testing. 
 Specifically, we aim to `1)` identify scenarios where our approach fails, `2)` create functions to accomodate these cases, and `3)` compare software results to previous manual approaches for extracting quantitative information, especially for pillar tracking. We will continue to update this repository as the project progresses.

## Installation Instructions <a name="install"></a>

### Get a copy of the microbundle compute repository on your local machine

The best way to do this is to create a GitHub account and ``clone`` the repository. However, you can also download the repository by clicking the green ``Code`` button and selecting ``Download ZIP``. Download and unzip the ``MicroBundleCompute-master`` folder and place it in a convenient location on your computer.

Alternatively, you can run the following command in a ``Terminal`` session:
```bash
git clone https://github.com/elejeune11/MicroBundleCompute.git
```
Following this step, ``MicroBundleCompute`` folder will be downloaded in your ``Terminal`` directory. 

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

1. Use a ``Terminal`` session to navigate to the ``MicroBundleCompute-master`` folder or ``MicroBundleCompute`` folder (depending on the method you followed to download the github repository). The command ``cd`` will allow you to do this (see: [terminal cheat sheet](https://terminalcheatsheet.com/))
2. Type the command ``ls`` and make sure that the file ``pyproject.toml`` is in the current directory.
3. Now, create an editable install of microbundle compute:
```bash
pip install -e .
```
4. If you would like to see what packages were installed, you can type ``pip list``
5. You can test that the code is working with pytest (all tests should pass):
```bash
pytest -v --cov=microbundlecompute  --cov-report term-missing
```
6. To run the code from the terminal, simply start python (type ``python``) and then type ``from microbundlecompute import image_analysis as ia``. For example:
```bash
(microbundle-compute-env) hibakobeissi@Hibas-MacBook-Pro ~ % python
Python 3.9.13 (main, Oct 13 2022, 16:12:19) 
[Clang 12.0.0 ] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> from microbundlecompute import image_analysis as ia
>>> ia.hello_microbundle_compute()
'Hello World!'
>>> 
```

## Tutorial <a name="tutorial"></a>
This GitHub repository contains a folder called ``tutorials`` that contains an example dataset and a python script for running the code. To run the tutorials, change your curent working directory to the ``tutorials`` folder.

The data (frames to be tracked) will be contained in the ``movie`` folder. Critically:
1. The files must have a ``.TIF`` extension.
2. The files can have any name, but in order for the code to work properly they must be *in order*. For reference, we use ``sort`` to order file names. By default, this function sorts strings (such as file names) alphabetically and numbers numerically. We provide below examples of good and bad file naming practices. 

```bash
(microbundle-compute-env) hibakobeissi@Hibas-MacBook-Pro MicroBundleCompute-master % python
Python 3.9.13 (main, Oct 13 2022, 16:12:19) 
[Clang 12.0.0 ] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> bad_example = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"]
>>> bad_example.sort()
>>> print(bad_example)
['1', '10', '11', '12', '13', '14', '15', '2', '3', '4', '5', '6', '7', '8', '9']
>>> 
>>> good_example = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15"]
>>> good_example.sort()
>>> print(good_example)
['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
>>>
>>> another_good_example = ["test_001","test_002","test_003","test_004","test_005","test_006","test_007","test_008","test_009","test_010","test_011","test_012","test_013","test_014","test_015"]
>>> another_good_example.sort()
>>> print(another_good_example)
['test_001', 'test_002', 'test_003', 'test_004', 'test_005', 'test_006', 'test_007', 'test_008', 'test_009', 'test_010', 'test_011', 'test_012', 'test_013', 'test_014', 'test_015']
```

3. If it is necessary to read other file types or formats (e.g., a single 3D TIFF array), that would be easy to implement -- single images are implemented now so that we don't exceed maximum file sizes allowable on GitHub.


All masks, whether for the tissue or the pillars, will be contained in the ``masks`` folder. Critically:
1. The current version of the code can use externally generated masks titled ``tissue_mask.txt`` for the tissue and ``pillar_mask_1.txt`` and ``pillar_mask_2.txt`` for the pillars. We note here that if the user wishes to track one pillar only, it is enough to provide either ``pillar_mask_1.txt`` or ``pillar_mask_2.txt``.
3. Future functionality for new mask generation functions is possible.
4. In our examples, ``tissue_mask.png``, ``pillar_mask_1.png``, and ``pillar_mask_1=2.png`` are simply visualizations of the corresponding mask text files. They are not necessary to run the code.

For the code to work properly, we provide below an example of the initial folder structure if both tissue and pillar tracking are to be run. Alternatively, if only one option is chosen, the corresponding mask(s) only is(are) required to be contained in the ``masks`` folder.
```bash
|___ example_folder
|        |___ movie
|                |___"*.TIF"
|        |___ masks    (this folder can be omitted if automated mask generation will be run as a first step and tissue tracking is run only. It is crucial to have this folder if pillar tracking is chosen)
|                |___"tissue_mask.txt"
|                |___"tissue_mask.png"        (optional)
|                |___"pillar_mask_1.txt"      
|                |___"pillar_mask_1.png"      (optional)
|                |___"pillar_mask_2.txt"
|                |___"pillar_mask_2.png"      (optional)

### Current core functionalities
In the tutorial provided, there are 5 core functionalities available. As a brief note, it is: (1) not necessary to use all functionality (e.g., you can consider displacement but ignore strain or skip the visualization steps), (2) additional functionality is currently under development, and (3) for the code snippets in this section the variable ``folder_path`` is a [``PosixPath``](https://docs.python.org/3/library/pathlib.html) that specifies the relative path between where the code is being run and the ``example_folder`` defined [above](#data_prep).
## Validation <a name="validation"></a>


## To-Do List <a name="todo"></a>
- [ ] Expand the test example dataset
- [ ] Compare pillar tracking functionality to tools available in the literature
- [ ] Extend the software capabilities to include tracking of calcium images
- [ ] Explore options for additional analysis/visualization


## References to Related Work <a name="references"></a>
Related work can be found here:
* Das, S. L., Sutherland, B. P., Lejeune, E., Eyckmans, J., & Chen, C. S. (2022). Mechanical response of cardiac microtissues to acute localized injury. American Journal of Physiology-Heart and Circulatory Physiology, 323(4), H738-H748.

Related repositories include:
* https://github.com/elejeune11/Das-manuscript-2022
* https://github.com/HibaKob/SyntheticMicroBundle (synthetic dataset)
* https://github.com/elejeune11/MicroBundleCompute-Lite (deprecated version of the code)

## Contact Information <a name="contact"></a>
For additional information, please contact Emma Lejeune ``elejeune@bu.edu`` or Hiba Kobeissi ``hibakob@bu.edu``.

## Acknowledgements <a name="acknowledge"></a>
Thank you to Shoshana Das for providing the example tissue included with this repository. And -- thank you to Chad Hovey for providing templates for I/O, testing, and installation via the [Sandia Injury Biomechanics Laboratory](https://github.com/sandialabs/sibl) repository.

