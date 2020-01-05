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
#Convolutional layer
class ConvolutionLayer(nn.Module):
    '''
    Args:
    in_channels = the number of channels of input channel (type int)
    output_channels = the number of output feature maps (type int)
    kernel_size = the size of the kernel (type tuple)
    strides = the length of the strides (type int)
    padding = takes two 1) valid, 2) same or some other integer value [default:"valid"]
    
    '''
    def __init__(self,in_channels,out_channels,kernel_size=(3,3),strides=1,padding="valid"):
        super(ConvolutionLayer,self).__init__()
        
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.kernel_size = kernel_size
        self.padding     = padding
        self.strides     = strides
        #layer definition
        self.convolution_layer = self.conv_layer(in_channels,out_channels,kernel_size,strides,padding)
        self.batch_normalize = nn.BatchNorm2d(out_channels)
        self.leaky_relu       = nn.LeakyReLU()
        

        
    def forward(self,x):
        x = self.convolution_layer(x)
        x = self.leaky_relu(x)
        x = self.batch_normalize(x)
        return x
    
    def conv_layer(self,in_channels,out_channels,kernel_size,strides,padding):
        if padding == "valid":
            padding =0
        elif padding == "same":
            strides = 1
            padding = math.floor(int((kernel_size[0]-1)/2))  
        return nn.Conv2d(in_channels,out_channels,kernel_size,strides,padding)

    
    
class LstmCell(nn.Module):
    '''
    Args:
    input_dims : the input dimension (that takes single integer) (type int)
    hidden dims : the dimension of hidden and cell state (type int)
    attach_fc  : to ensure whether fully connected layer should be connected to last rnn cell [default:False]
    '''
    def __init__(self,input_dims,hidden_dims,attach_fc = False):
        super(LstmCell,self).__init__()
        
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.attach_fc = attach_fc
        
        self.lstm_cell = nn.LSTMCell(input_size = self.input_dims,hidden_size = self.hidden_dims)
        if self.attach_fc == True:
            self.fc = nn.Linear(self.hidden_dims,1)
        
    def forward(self,x,hidden_state,cell_state):
        hidden_output,cell_output = self.lstm_cell(x,(hidden_state,cell_state))
        if self.attach_fc ==True:
            output = self.fc(hidden_output)
            return output
        else:
            return hidden_output,cell_output
        
#     def init_hidden(self, batch_size):
#         hidden = torch.tensor(next(self.parameters()).data.new(batch_size, self.hidden_dims), requires_grad=False)
#         cell = torch.tensor(next(self.parameters()).data.new(batch_size, self.hidden_dims), requires_grad=False)
#         return hidden.zero_(), cell.zero_()
class VRNet(nn.Module):
    def __init__(self,in_channels,cnn_layer,lstm_cell):
        super(VRNet,self).__init__()
        self.hidden_dims =128
        self.input_dims =128
        self.stage_lstm =lstm_cell(self.input_dims,self.hidden_dims)
        #stage1
        self.stage1_cnn = cnn_layer(in_channels=3,out_channels=32,kernel_size=(3,3),padding="valid",strides=1)
        self.stage1_pool = nn.AvgPool2d(kernel_size=(3,3),stride=2)
        #stage1 inter
        self.stage1_inter_cnn3x3 = cnn_layer(in_channels=32,out_channels=64,kernel_size=(3,3),padding="valid",strides=1)
        self.stage1_inter_cnn1x1 = cnn_layer(in_channels=64,out_channels=128,kernel_size=(1,1),padding="valid",strides=1)
        self.stage1_interpool = nn.AdaptiveAvgPool2d(1)
       

        #stage2
        self.stage2_cnn = cnn_layer(in_channels=32,out_channels=64,kernel_size=(3,3),padding="same",strides=1)
        self.stage2_pool = nn.AvgPool2d(kernel_size=(3,3),stride=2)
        #stage2 inter
        self.stage2_inter_cnn3x3 = cnn_layer(in_channels=64,out_channels=64,kernel_size=(3,3),padding="valid",strides=1)
        self.stage2_inter_cnn1x1 = cnn_layer(in_channels=64,out_channels=128,kernel_size=(1,1))
        self.stage2_interpool = nn.AdaptiveAvgPool2d(1)
        
        #stage3
        self.stage3_cnn = cnn_layer(in_channels=64,out_channels=64,kernel_size=(3,3),padding="valid")
        self.stage3_pool = nn.AvgPool2d(kernel_size=(3,3),stride=1)
        #stage3 inter
        self.stage3_inter_cnn3x3 = cnn_layer(in_channels=64,out_channels=64,kernel_size=(3,3),padding="valid",strides=1)
        self.stage3_inter_cnn1x1 = cnn_layer(in_channels=64,out_channels=128,kernel_size=(1,1))
        self.stage3_interpool = nn.AdaptiveAvgPool2d(1)
        #linear
        self.fc = nn.Linear(self.hidden_dims,1)
        
        
    def forward(self,x,hidden_state,cell_state):
        #stage1
        x = self.stage1_cnn(x)
        x = self.stage1_pool(x)
        #stage1 inter
        x1 = self.stage1_inter_cnn3x3(x)
        x1 = self.stage1_inter_cnn1x1(x1)
        x1  = self.stage1_interpool(x1)
        x1 = x1.squeeze()
        hidden_state,cell_state = self.stage_lstm(x1,hidden_state,cell_state)
        del x1
        
        #stage2
        x = self.stage2_cnn(x)
        x = self.stage2_pool(x)
        #stage2 inter
        x1 = self.stage2_inter_cnn3x3(x)
        x1 = self.stage2_inter_cnn1x1(x1)
        x1 = self.stage2_interpool(x1)

        x1 = x1.squeeze()
        hidden_state,cell_state = self.stage_lstm(x1,hidden_state,cell_state)
        del x1
        #stage3
        x = self.stage3_cnn(x)
        x = self.stage3_pool(x)
        #stage3 inter
        x = self.stage3_inter_cnn3x3(x)
        x = self.stage3_inter_cnn1x1(x)
        x = self.stage3_interpool(x)
        x = x.squeeze()
        hidden_state,cell_state = self.stage_lstm(x,hidden_state,cell_state)
        del x,cell_state
        return self.fc(hidden_state)
        
    def init_hidden(self, batch_size):
        hidden = torch.tensor(next(self.parameters()).data.new(batch_size, self.hidden_dims), requires_grad=False)
        cell = torch.tensor(next(self.parameters()).data.new(batch_size, self.hidden_dims), requires_grad=False)
        return hidden.zero_(), cell.zero_()
        
    
