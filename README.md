# MONet - SAR Despceckling CNN - (python/theano implementation)

This the python (Theano) implementation of the testing phase for **MONet**, a CNN for SAR despeckling described in the paper
[Multi-Objective CNN Based Algorithm for SAR
Despeckling](https://ieeexplore.ieee.org/document/9261137).

MONet is 17 layers CNN with skip connection and a multi-objective cost function L. L is composed of three terms: MSE between output and reference, Kullback-Leibler divergence between estimated noise distribution and the theoretical one, and an edge loss computed on output and the reference. The architecture is shown in the following

![net](/samples/net-1.png)

This a an example of the results on data with simulated speckle
 Noisy Image| Noise-Free Reference | MONet - Output 
-----------------------------------------|---------------------------------------|--------------------------------------
![img1](/samples/storagetanks_noisy.png) |![img2](/samples/storagetanks_ref.png) |![img3](/samples/storagetanks_out.png)

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
This code is written on Ubuntu system for Python2.7 and uses Theano library.

The command to install the requirements is: 

```
cat requirements.txt | xargs -n 1 -L 1 pip2 install
```

### Anaconda (Optional)
If you use a python editor
* install [Anaconda](https://repo.anaconda.com/archive/)
* install requirements and **spyder** editor with **conda**
* edit **main.py** and run

### Optional requirements for using gpu:
* cuda = 8 
* cudnn = 5


# Usage 
* **imgs** folder contains three samples images with simulated single look speckle in amplitude format;
the sample image are taken from [UC Merced LandUse Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html).
Three differente <AREA> can be tested:
     * baseballdiamond
     * golfcourse
     * storagetanks

* **model** folder contains the pre-trained network
* run test without GPU
```
python main.py -a <AREA>
```
* run with GPU
```
PATH=<CUDAPATH>:$PATH python main.py -g -a <AREA>
```
