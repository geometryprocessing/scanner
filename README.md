# [Siggraph Asia 2025] LookUp3D: Data-Driven 3D Scanning

### [Paper](https://arxiv.org/abs/2405.14882) | [Data](https://doi.org/10.58153/1n4p2-ygf77)

![Bunny in Free Fall at 450 FPS](media/teaser.png)

<strong>Giancarlo Pereira*<sup>2</sup>, Yidan Gao*<sup>1</sup>, Yurii Piadyk*<sup>2</sup>, David Fouhey<sup>1,2</sup>, Claudio T. Silva<sup>2,3</sup>, Daniele Panozzo<sup>1</sup></strong>

<small>*Joint authors with equal contribution</small>

<small><sup>1</sup>New York University, Courant Institute of Mathematical Sciences</small>

<small><sup>2</sup>New York University, Tandon School of Engineering</small>

<small><sup>3</sup>New York University, Center for Data Science</small>

This repository contains the main algorithmic implementations of our work ["LookUp3D: Data-Driven 3D Scanning"](https://arxiv.org/abs/2405.14882).

For data (such as the reconstructed point clouds of our paper or to demo how our scanning works), please see our [NYU UltraViolet repository](https://doi.org/10.58153/1n4p2-ygf77).

For scripts on controlling hardware, please see [this repository](https://github.com/geometryprocessing/scanner-capture/). It also contains the firmware for controlling the high-speed analog projector we developed for LookUp3D.

## Installation
This library has been built with python 3.12 and tested on MacOS and Linux Ubuntu 20.04 and 24.04.

We recommmend using a virtual environment to manage libraries and avoid dependency conflicts. For example, with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html):
```
conda env create -n scanner python=3.12
conda activate scanner
pip install -r requirements.txt
```

### CUDA Implementation (NEW FEATURE, NOT USED IN PAPER):
If you would like to make use of the parallel reduction implementation with CUDA, please install:
```
pip install numba-cuda cupy
```
For now we use cupy as its syntax allows for interoperability with numpy.

## Acknowledgments
This work was partially supported by the the NSF grants OAC-2411349 and OAC-2411221. Giancarlo Pereira was partially supported by the New York University Tandon School of Engineering Fellowship.

We thank NYU IT High Performance Computing services, for help with resources, services, and expertise. We also would like to thank [Professor Christopher Musco](https://www.chrismusco.com) for fruitful discussions on low-rank approximation and would like to thank [Arvi Gjoka](https://www.arvigjoka.com) for making a silicone bunny.


## Citation
If you use this work/data, please be kind to cite our paper:

	@inproceedings{10.1145/3757377.3763986,
    author = {Pereira, Giancarlo and Gao, Yidan and Piadyk, Yurii and Fouhey, David and Silva, Claudio T and Panozzo, Daniele},
    title = {LookUp3D: Data-Driven 3D Scanning},
    year = {2025},
    isbn = {9798400721373},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3757377.3763986},
    doi = {10.1145/3757377.3763986},
    booktitle = {Proceedings of the SIGGRAPH Asia 2025 Conference Papers},
    articleno = {149},
    numpages = {11},
    keywords = {3D Scanning, Geometry Acquisition, Structured Light, Data-Driven, Active Illumination, High-Speed},
    location = {
    },
    series = {SA Conference Papers '25}
    }