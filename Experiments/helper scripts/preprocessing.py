import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage import io,transform
import torch
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import os
import time
from torch.optim import lr_scheduler
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils,models
import copy
import math
from skimage import io, transform
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class AsetheticsDataset(Dataset):
      '''asethitics dataset'''
      def __init__(self,dataframe,root_dir,transform=None):
        """
            Args:
                csv_file (string): Path to the csv file with annotations.
                root_dir (string): Directory with all the images.
                transform (callable, optional): Optional transform to be applied
                    on a sample.
        """
        self.data = dataframe
        #     self.data.rename(columns=columns,inplace=True)
    #     self.data.drop(self.data.columns[[1,2,3,4,5,6,8,9]] , axis=1,inplace=True)
        self.root_dir = root_dir
        self.transform = transform
    
      def __len__(self):
        return len(self.data)
  
      def __getitem__(self,idx):
   
        if torch.is_tensor(idx):
          idx = idx.tolist()
   
        image_name =  os.path.join(self.root_dir,self.data.iloc[idx,0])
        image = io.imread(image_name)
        mem_val = self.data.iloc[idx,1]
#     return_sample={}
        return_sample = {
              'image':image,
              'memorability_score':mem_val 
        }
        if self.transform:
            return_sample = self.transform(return_sample)
    
     
        return return_sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        
        image,mem_val = sample['image'], sample["memorability_score"]
        
        h, w = image.shape[:2]
        
        # if isinstance(self.output_size, int):
        #     if h > w:
        #         new_h, new_w = self.output_size * h / w, self.output_size
        #     else:
        #         new_h, new_w = self.output_size, self.output_size * w / h
        # else:
        #     new_h, new_w = self.output_size

        # new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (self.output_size,self.output_size,3))
        return {'image': img, 'memorability_score': mem_val}
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        image, mem_val = sample['image'], sample['memorability_score']
#         print(type(torch.from_numpy(image)))
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
#         print(image.shape)
      
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'memorability_score': mem_val}
class Normalize(object):
    def __init__(self,mean,std):
        self.mean=mean
        self.std=std
    def __call__(self,sample):
        image, mem_val = sample["image"], sample["memorability_score"]
        normalized=  (image -self.mean) / self.std
        return {
            "image":normalized,
            "memorability_score" : mem_val
       }

# transformed_dataset_train = AsetheticsDataset(dataset_train,root_dir="/content/lamem/images",
#                                         transform=transforms.Compose([Rescale(224),ToTensor(),Normalize(0.5,0.5)
#                                                           ]))

# transformed_dataset_val= AsetheticsDataset(dataset_validation,root_dir="/content/lamem/images",
#                                         transform=transforms.Compose([Rescale(224),ToTensor(),Normalize(0.5,0.5)
#                                                           ]))

# train_dataloader=DataLoader(transformed_dataset_train,batch_size=32,shuffle=True)
# validation_dataloader=DataLoader(transformed_dataset_val,batch_size=32,shuffle=True)
# dataloaders={
#     "train":train_dataloader,
#     "val":validation_dataloader
# }
# dataset_sizes ={
#     "train":len(dataset_train),
#     "val":len(dataset_validation)
# }