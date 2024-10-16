# `tatm` Kempner AI Testbed Library

`tatm` (**T**ransformer **A**ssistive **T**estbed **M**odule) is a Python library that provides a set of tools to assist in the development of transformer-based (and other architecture) models for research within the Kempner institute. It will provide an interface for accessing and manipulating data, training models, and evaluating models. The library is designed to be modular and extensible, allowing for easy integration of new models, datasets, and training frameworks.

## Documentation

[![Documentation Status](https://readthedocs.org/projects/tatm/badge/?version=latest)](https://tatm.readthedocs.io/en/latest/?badge=latest)x

- [Stable Version Documentation](https://tatm.readthedocs.io/en/latest/index.html)
   - [Getting Started](https://tatm.readthedocs.io/en/latest/getting_started.html)
   - [Getting Started](https://tatm.readthedocs.io/en/latest/text_dataset.html)

- [Nightly Version Documentaiton](https://kempnerinstitute.github.io/tatm/)
   - [Getting Started](https://kempnerinstitute.github.io/tatm/getting_started.html)
   - [Example Working with a Text Dataset](https://kempnerinstitute.github.io/tatm/text_dataset.html)

## Installation

### Requirements

- Python 3.10 is the minimum supported version. This is in order to match the standard version of python available on the 
   Cannon cluster.

- `tatm` depends on pytorch, which depends on CUDA. It is recommended to pre-install both pytorch and CUDA prior to installing `tatm`. Instructions for installing pytorch can be found [here](https://pytorch.org/get-started/locally/).

### Installation

#### Installing from GitHub

To install the latest stable version of `tatm` from GitHub, run the following command:

```bash
pip install git+ssh://git@github.com/KempnerInstitute/tatm.git@main
```

To install the latest development version of `tatm` from GitHub, run the following command:

```bash
pip install git+ssh://git@github.com/KempnerInstitute/tatm.git@dev
```

For a specific past version of `tatm`, replace `main` or `dev` with the desired version number (i.e. `v0.1.0`).

#### Installing from PyPI

The package is not yet available on PyPI. Stay tuned for updates!