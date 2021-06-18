# MONet--Analysis-on-the-Building-of-Training-Datatset-for-Deep-Learning-SAR-Despeckling - (python/pytorch implementation)

This code is related to the paper 
**Analysis-on-the-Building-of-Training-Datatset-for-Deep-Learning-SAR-Despeckling** accepted for Geascience oand Remote Sensing Letters

The pytorch model of **MONet** trained with the three approaches indicated in the paper are provided togheter with the testing phase.

[MONet](https://ieeexplore.ieee.org/document/9261137) is 17 layers CNN with skip connection and a multi-objective cost function L. L is composed of three terms: MSE between output and reference, Kullback-Leibler divergence between estimated noise distribution and the theoretical one, and an edge loss computed on output and the reference. The architecture is shown in the following

![net](/samples/net-1.png)

In the folder './models/' three trained model of **MONet** are provided:
 1. synthetic_MONet: it is the MONet architecture trained with synthetic approach
 2. multitemporal_MONet: it is the MONet architecture trained with the multitemporal approach
 3. hybrid_MONet: it is the MONet architecture trained with the hybrid approach
 

# Team members
 Sergio Vitale    (contact person, sergio.vitale@uniparthenope.it);
 Giampaolo Ferraioli (giampaolo.ferraioli@uniparthenope.it);
 Vito Pascazio (vito.pascazio@uniparthenope.it)
 
# License
Copyright (c) 2020 Dipartimento di Ingegneria and Dipartimento di Scienze e Tecnologie of Università degli Studi di Napoli "Parthenope".

All rights reserved. This work should only be used for nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this directory)

# Prerequisits
This code is written on Ubuntu system for Python3.7 and uses Pytorch library.

The requirements and installation instruction are in the **requiremetnts.txt** file


# Usage 
* edit and run **test.py**
*  **imgs** folder contains one real SAR sample from the ICEYE sensors.
* The image is a  single look speckle in amplitude format;

* **models** folder contains the pre-trained networks
