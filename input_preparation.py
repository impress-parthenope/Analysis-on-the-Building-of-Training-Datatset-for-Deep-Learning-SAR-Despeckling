#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:01:49 2021

@author: sergio
"""
import numpy as np
import sys
eps = sys.float_info.epsilon
import torch

def net_scope(net):
    "compute the scope of the network"
    blk = 0
    for l in net.named_children():
        if len(l[1].weight.shape)==4:
            blk+=np.floor(l[1].weight.shape[-1]/2)
    
    return np.int(blk)

def preparation(I,blk):

    I_in = np.where(np.equal(I,0),eps,I) #exceptions
    I_in = np.pad(I_in, ((blk,blk),(blk,blk)),mode='edge') #input padding
    
    return torch.from_numpy(I_in[np.newaxis,np.newaxis,:,:])