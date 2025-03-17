# Modeling individual mental representations underlying similarity judgments

Code accompanying the following paper: https://doi.org/10.31234/osf.io/agpb5_v1

This repository contains a python package to fit models on data from the odd-one-out task as described in the paper. 

In [paper/](paper/) you can find the code and data to reproduce the results from the paper (the speaker odd-one-out task).

## Installation

Dependencies are included in the [environment.yml](environment.yml) file and can be installed in a conda environment 

```
conda env create -f environment.yml
```

You can then install the package with 
```
pip install .
```

and run the test script in [tests/](tests/) to ensure the installation is working.

## Usage

Please refer to the Jupyter notebooks in [examples/](examples/) for demonstrations of how to model odd-one-out choices using our approach.
