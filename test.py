"""
This is the testing code for the three trained model of MONet related ot the accpeted paper:
    Vitale S., Ferraioli G., Pascazio V., 
    "Analysis on the Building of Training Datatset for Deep Learning SAR Despeckling"
    Geoscience and Remote Sensing Letters, 2021
    
If you find useful this code, cite it as: <CITAZIONE>

"""
"""
This code test on a real SAR image the MONet architecture trained with three different approaches:
    1. Synthetic
    2. Multitemporal
    3. Hybrid

In the section PATH SETTING, you can find and change the path for testing model and images

# path_in:          path of the input testing image
# img:              name of the testing file
# path_out:         path in which saving the results
# modelpath:        path of trained models
# train_dataset:    list of three trained approaches
# device:           select the processing device: 
                                    set device =-1 for cpu,
                                    set device = <x> for using the gpu number <x> 
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import os
import sys
import gc
sys.path.append('./Utilities')
eps = sys.float_info.epsilon

#%% PATH SETTING

path_in = './imgs/'         #testing image path
img ='y_singapore_airport'  #testing file name
path_out = './results/'     #saving folder
model_path = './models/'    #trained model path
train_dataset = ['synthetic','multitemporal','hybrid']
device=-1           # device selection


if device >=0:
    device = torch.device('cuda:%d'%(device) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

if not os.path.exists(path_out):
    os.makedirs(path_out)

#%% Testing for each training dataset
from model_MONet import Net
from input_preparation import net_scope, preparation

net=Net() #model loading
blk = net_scope(net) # scope of the network

for dataset in train_dataset:
    print('Testing MONet trained with '+dataset+' dataset')
    #Loading Trained Model
    net.load_state_dict(torch.load(model_path+dataset+'_MONet'))
    net.eval()
    net.to(device)
    
    #Testing
    with torch.no_grad():
        I =sio.loadmat(path_in+img+'.mat',squeeze_me=True)
        I = np.asarray(abs(I['noisy']),dtype='float32') #amplitude image
    
        #preparation for input
        I_in = preparation(I,blk)
        
        #Despeckling
        I_out= net(I_in.to(device))                   
        I_out = I_out.cpu().detach().numpy()
        #save 
        sio.savemat(path_out+img+'_'+dataset+'.mat',{'output':np.squeeze(I_out[0,0,:,:])})
    
#%% visualize results
plt.close('all')
    
I= sio.loadmat(path_in+img+'.mat',squeeze_me=True)
I = np.asarray(abs(I['noisy']),dtype='float32') #amplitude image
    
I_synt = sio.loadmat(path_out+img+'_synthetic.mat',squeeze_me=True)
I_multi = sio.loadmat(path_out+img+'_multitemporal.mat',squeeze_me=True)
I_hybrid = sio.loadmat(path_out+img+'_hybrid.mat',squeeze_me=True)

I_synt = I_synt['output']
I_multi = I_multi['output']
I_hybrid = I_hybrid['output']

I_synt_ratio = I/I_synt
I_multi_ratio = I/I_multi
I_hybrid_ratio = I/I_hybrid

plt.figure()
plt.subplot(241),plt.imshow(I[:3000,:3000],cmap='gray',vmin=0,vmax=125), plt.title('SAR')
plt.subplot(242),plt.imshow(I_synt[:3000,:3000],cmap='gray',vmin=0,vmax=125), plt.title('MONet-Synt')
plt.subplot(243),plt.imshow(I_multi[:3000,:3000],cmap='gray',vmin=0,vmax=125), plt.title('MONet-Multi')
plt.subplot(244),plt.imshow(I_hybrid[:3000,:3000],cmap='gray',vmin=0,vmax=125), plt.title('MONet-Hybrid')
plt.subplot(246),plt.imshow(I_synt_ratio[:3000,:3000],cmap='gray',vmin=0,vmax=2.5), plt.title('Ratio-Synth')
plt.subplot(247),plt.imshow(I_multi_ratio[:3000,:3000],cmap='gray',vmin=0,vmax=2.5), plt.title('Ratio-Multi')
plt.subplot(248),plt.imshow(I_hybrid_ratio[:3000,:3000],cmap='gray',vmin=0,vmax=2.5), plt.title('Ratio-Hybrid')

