import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import math
from utils import *

class myDataSet(Dataset):
    def __init__(self,root,nsplit, manifest, 
                 normalize=True,train=True,
                maxSeiz = 10
                 ):
        self.root = root+'/clean_data/'
        if train:
            ptlist = np.load(root+'split'+str(nsplit)+'/train_pts.npy')
        else:
            ptlist = np.load(root+'split'+str(nsplit)+'/val_pts.npy')
        self.mnlist = [mnitem for mnitem in manifest if json.loads(mnitem['pt_id']) in ptlist ]
        self.normalize = normalize
        self.nchn = 19
        self.maxSeiz = maxSeiz 
 
    def __getitem__(self, idx):

        mnitem = self.mnlist[idx]
        fn = mnitem['fn']
        pt = int(mnitem['pt_id'])
        isnoisy = False
        xloc = self.root+fn
        yloc = xloc.split('.')[0] + '_label.npy'

        X = np.load(xloc)[:self.maxSeiz, :,:,:]
        Y = np.load(yloc)[:self.maxSeiz]
        soz = self.load_onset_map(mnitem)
        if self.normalize:
            X = (X - np.mean(X))/np.std(X)
            
        noise_labels =  []
        xhat = []
      
        return {'patient numbers': pt,
                'buffers':torch.Tensor(X), #original
                'sz_labels':torch.Tensor(Y),  #sz
                'onset map':torch.Tensor(soz), #soz
                'isnoisy': isnoisy
               }  
    
    def load_onset_map(self, mnitem):
        req_chn = ['fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8', 't3', 'c3', 'cz', 
                    'c4', 't4', 't5', 'p3', 'pz', 'p4', 't6', 'o1', 'o2']
        soz = np.zeros(len(req_chn))
        for i,chn in enumerate(req_chn):
            if mnitem[chn] != '':
                soz[i] = json.loads(mnitem[chn])
           
        return soz
    
    def __len__(self):                       #gives number of recordings
        return len(self.mnlist)