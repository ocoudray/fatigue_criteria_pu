# Construction of fatigue criteria through Positive Unlabeled Learning

This repository contains the code and data to reproduce the results of the paper entitled *Construction of fatigue criteria through Positive Unlabeled Learning* (Olivier Coudray, [Philippe Bristiel](https://www.linkedin.com/in/philippe-bristiel-8aa8755a/?originalSubdomain=fr), Miguel Dinis, [Christine Keribin](https://wp.imo.universite-paris-saclay.fr/christine-keribin/), [Patrick Pamphile](https://www.linkedin.com/in/patrick-pamphile-993830a5/?originalSubdomain=fr)).

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
git clone https://github.com/ocoudray/fatigue_criteria_pu.git
```
+ Enter the project directory
```bash
cd fatigue_criteria_pu
```
+ Install the dependencies using poetry
```bash
poetry install
```
**Important:** This installation is crucial. In particular, the content of "src" is installed as a dependency in the environment enabling import syntax of the form:
```python
from pu_criterion.PU import PUClassifier
```

## Organixation of the repository

The repository is organized as follows:
+ the folder [`data`](https://github.com/ocoudray/fatigue_criteria_pu/tree/main/data) contains the data used in the paper: data from Fayard thesis (coupon specimen) for the calibration of probabilistic Dang Van criterion and the fatigue database consisting of zones of complex automotive mechanical parts
+ the folder [`experiments`](https://github.com/ocoudray/fatigue_criteria_pu/tree/main/experiments) contains the results of the experiments carried out in section 5 of the paper. The notebook [`2.2.Experiments.ipynb`](https://github.com/ocoudray/fatigue_criteria_pu/blob/main/2.2.Experiments.ipynb) contains the code to re-run these experiments.
+ the folder [`figures`](https://github.com/ocoudray/fatigue_criteria_pu/tree/main/figures) contains the figures generated in the different notebooks. They can be reproduced by re-running the notebooks.
+ the folder [`src`](https://github.com/ocoudray/fatigue_criteria_pu/tree/main/src) contains the backend of this project. The main classes and functions are coded there and called in the notebooks.

## Reproducing the results of the paper

The results of the paper can be reproduced by running the different notebooks available at the root of this project. To do so, first launch jupyter and open the notebooks (do not forget to set up the correct kernel).

The notebooks are organized as follows:
1. [`1.Probabilistic_DV_criterion.ipynb`](https://github.com/ocoudray/fatigue_criteria_pu/blob/main/1.Probabilistic_DV_criterion.ipynb): this notebook implements the probabilistic Dang Van criterion and its estimation on Fayard coupon specimen. The related results and figures of the paper can be reproduced.
2. Results regarding the construction of fatigue criteria through PU learning:
    + [`2.1.PU_EM.ipynb`](https://github.com/ocoudray/fatigue_criteria_pu/blob/main/2.1.PU_EM.ipynb): this notebook can be used to launch a single experiment and presents the training/testing pipeline
    + [`2.2.Experiments.ipynb`](https://github.com/ocoudray/fatigue_criteria_pu/blob/main/2.2.Experiments.ipynb): this notebook allows to launch several experiments with randomized train/test and save the results. *Note that the absolute path to the data needs to be indicated in the first cell of the notebook*.
    + [`2.3.PU_learning_fatigue_results`](https://github.com/ocoudray/fatigue_criteria_pu/blob/main/2.3.PU_learning_fatigue_results.ipynb): this last notebook contains the code related to the analysis of the results. The results and figures of the paper (section 5)