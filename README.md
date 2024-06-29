# Construction of fatigue criteria through Positive Unlabeled Learning

This repository contains the code and data to reproduce the results of the paper entitled *Construction of fatigue criteria through Positive Unlabeled Learning* (Olivier Coudray, Philippe Bristiel, Miguel Dinis, Christine Keribin, Patrick Pamphile).

## Installation

The code is written in python. The project relies on poetry for dependencies management.

**Prerequisites:**
+ a version of python between 3.9 and 3.10 installed with jupyter
+ poetry installed
+ git installed

**Installation:**
+ Create an empty python environment (python 3.9 or 3.10), e.g. using venv:
```bash
python -m venv pul
```
+ Activate the new environnement, e.g. using venv on mac:
```bash
source ./pul/bin/activate
```
+ Clone this git repository
```bash
git clone ...
```
+ Enter the project directory
```bash
cd ...
```
+ Install the dependencies using poetry
```bash
poetry install
```
**Important:** This installation is crucial. In particular, the content of "src" is installed as a dependency in the environment enabling import syntax of the form:
```python
from pu_criterion.PU import PUClassifier
```

## Reproducing the results of the paper

The results of the paper can be reproduced by running the different notebooks available at the root of this project. To do so, first launch jupyter and open the notebooks (do not forget to set up the correct kernel).

The notebooks are organized as follows:
1. `1_Probabilistic_DV_criterion.ipynb`: this notebook implements the probabilistic Dang Van criterion and its estimation on Fayard coupon specimen. The related results and figures of the paper can be reproduced.
2. Results regarding the construction of fatigue criteria through PU learning:
    + `2_1.PU_EM.ipynb`: this notebook can be used to launch a single experiment and presents the training/testing pipeline
    + `2_2.Experiments.ipynb`: this notebook allows to launch several experiments with randomized train/test and save the results. *Note that the absolute path to the data needs to be indicated in the first cell of the notebook*.
    + `2_3.PU_learning_fatigue_results`: this last notebook contains the code related to the analysis of the results. The results and figures of the paper (section 5)