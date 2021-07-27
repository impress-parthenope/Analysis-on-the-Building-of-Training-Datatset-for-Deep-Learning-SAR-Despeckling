# Analysis on the Building of Training Datatset for Deep Learning SAR-Despeckling - (python/pytorch implementation)

This code is related to the paper 
[**Analysis on the Building of Training Datatset for Deep Learning SAR Despeckling**](https://ieeexplore.ieee.org/document/9474572) published on Geoscience and Remote Sensing Letters.

The pytorch model of **MONet** trained with the three approaches indicated in the paper are provided togheter with the testing phase.

[MONet](https://ieeexplore.ieee.org/document/9261137) is 17 layers CNN with skip connection and a multi-objective cost function L. L is composed of three terms: MSE between output and reference, Kullback-Leibler divergence between estimated noise distribution and the theoretical one, and an edge loss computed on output and the reference. The architecture is shown in the following

![monet](https://user-images.githubusercontent.com/85936968/122596576-5d93a500-d06a-11eb-8ab3-90e44a7d61a2.PNG)

The pytorch MONet architecture is defined in **'./Utilities/model_MONET.py'**

In the folder './models/' three trained models of **MONet** are provided:
 1. **synthetic_MONet**: MONet architecture trained with synthetic approach (this is the modelpublished in [MONet](https://ieeexplore.ieee.org/document/9261137))
 2. **multitemporal_MONet**: MONet architecture trained with the multitemporal approach
 3. **hybrid_MONet**: MONet architecture trained with the hybrid approach
 
For more details on the synthetic, multitemporal and hybrid apporach, please refer to the paper.

# Team members
 Sergio Vitale    (contact person, sergio.vitale@uniparthenope.it);
 Giampaolo Ferraioli (giampaolo.ferraioli@uniparthenope.it);
 Vito Pascazio (vito.pascazio@uniparthenope.it)
 
# License
Copyright (c) 2021 Dipartimento di Ingegneria and Dipartimento di Scienze e Tecnologie of Università degli Studi di Napoli "Parthenope".

All rights reserved. This work should only be used for nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this directory)

# Prerequisites
This code is written on Ubuntu system for Python3.7 and uses Pytorch library.

For a correct usage of the code, please install the python environement saved in **./env/monet_pytorch.yml** with the following step:

**Installing Anaconda** (if not already installed)

1. download anaconda3 from https://www.anaconda.com/products/individual#linux
2. from command line, move to the download directory and install the package by:
> sh <Anaconda_downloaded_version>.sh and follow the instruction for installation
3. add conda to path
> PATH=~/anaconda3/bin:$PATH

**Installing the conda environment**
1. the file ./env/monet_pytorch.yml contains the environemnt for the testing the code. You can easily installing it by command line:

> conda env create -f monet_pytorch.yml

# Usage 
* from command line: activate the environement monet_pytorch and run spyder
> conda activate monet_pytorch

> spyder
* edit and run **test.py**
*  **imgs** folder contains one real SAR sample from the ICEYE sensors.
* The image is a  single look speckle in amplitude format;

* **models** folder contains the pre-trained networks ready for testing
