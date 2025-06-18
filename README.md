# AudeyART

This is an educational repository for learning about ART algorithms, mainly through writing up a basic FuzzyART module and testing how it works on the Iris dataset.

## Table of Contents

- [AudeyART](#audeyart)
  - [Table of Contents](#table-of-contents)
  - [Usage](#usage)
  - [File Structure](#file-structure)
  - [Requirements Description](#requirements-description)

## Usage

First, get a good Python setup going.
I highly recommend using a virtual environment manager, such as [`mamba`](https://mamba.readthedocs.io/en/latest/index.html).
For example, download the [miniforge distribution](https://github.com/conda-forge/miniforge) for your OS.

Next, create a virtual environment like so:

```shell
mamba create -n audeyart python=3.12
```

Activate that environment with

```shell
mamba activate audeyart
```

and install the python dependencies once you're inside that environment with

```shell
pip install -r requirements.txt
```

> [!note]
> Read through the [Requirements Description](#requirements-description) to get an understanding of what each dependency is used for and get links to their respective documentations.

## File Structure

This section outlines the location and meaning of the files in this repo:

- [`data/`](data)
  - [`iris/`](data/iris): a download of the Iris flower dataset from [UCI machine learning data repository](https://archive.ics.uci.edu/dataset/53/iris).
    - [`bezdekIris.data`](data/iris/bezdekiris.data): a version of the data that presumably [Jim Bezdek](https://scholar.google.com/citations?user=kXy4LAMAAAAJ&hl=en).
    - [`Index`](data/iris/Index): some timestamps of versions of the dataset.
    - [`iris.data`](data/iris/iris.data): the dataset itself as a comma-separated values (CSV) file.
    - [`iris.names`](data/iris/iris.names): citations and descriptions of the elements of the dataset.
- [notebooks](notebooks)
  - [`audeyart.ipynb`](notebooks/audeyart.ipynb): an IPython notebook (a.k.a. jupyter notebook) containing the main pedagogical material.
  This includes:
      1. How to write up a FuzzyART module.
      2. How to load the Iris dataset with [`pandas`](https://pandas.pydata.org).
      3. How to cluster the Iris dataset with FuzzyART.
      4. How to visualize the results with [TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).
  - [`audeyart-original.ipynb`](notebooks/audeyart-original.ipynb): the original notebook used as an exercise for implementing FuzzyART (without added documentation).
  - [`supervised.ipynb`](notebooks/supervised.ipynb): an implementation of a simple FuzzyARTMAP (i.e., a simple supervised variant of FuzzyART).
- [`.flake8`](.flake8): some custom Python [Flake8](https://flake8.pycqa.org/en/latest/) linting preferences, such as the config to make `flake8` stop yelling if lines are longer than a meager 80 characters.
- [`.gitignore`](.gitignore): a file with patterns that are ignored by git tracking.
- [`LICENSE`](LICENSE): a text file containing the MIT license for the repo, indicating that this software is free for anyone to use in anyway with no liability attributable to the authors.
This is a common license for open-source software that allows other people to use and even profit from your work without being able to blame you when they break stuff.
- [`README.md`](README.md): this file.
Readme's are the most common top-level description of software repositories, and they are the best first place to describe your work to someone who would actually use it.
- [`requirements.txt`](requirements.txt): the pip requirements file containing all of the Python dependencies for the project.

## Requirements Description

The pip requirements under the `requirements.txt` file are listed below.
For convenience, the documentation for each is linked.

[pytorch-docs]: https://docs.pytorch.org/docs/stable/index.html
[torchvision-docs]: https://docs.pytorch.org/vision/stable/index.html
[matplotlib-docs]: https://matplotlib.org/stable/users/index
[jupyterlab-docs]: https://docs.jupyter.org/en/latest/
[pandas-docs]: https://pandas.pydata.org/docs/
[scikit-learn-docs]: https://scikit-learn.org/stable/

- [`torch`][pytorch-docs]: this is the pip name for PyTorch, one of the biggest libraries for working with tensor data and subsequently neural networks.
- [`torchvision`][torchvision-docs]: PyTorch's separate library for handling vision dataset pipelines, transformations, etc.
This wasn't used in the written example, but it is included because almost everything else we will write includes its functionality when working with bigger datasets.
- [`jupyterlab`][jupyterlab-docs]: an all-in-one dependency for installing an IPython kernel and the notebook environment.
- [`matplotlib`][matplotlib-docs]: the most comprehensive plotting library in Python.
- [`pandas`][pandas-docs]: provides the `DataFrame` datatype for Python and all of the utilities therein to load, parse, and handle tabular data.
- [`scikit-learn`][scikit-learn-docs]: the traditional machine learning toolset for Python.
Most of the _de facto_ machine learning techniques that aren't deep learning are already implemented here.
