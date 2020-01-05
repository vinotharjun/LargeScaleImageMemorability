import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.utils.tensorboard import SummaryWriter

#idenetity block
class IdentityBlock(nn.Module):
    def __init__(self,f,in_channels,filters):
        super(IdentityBlock,self).__init__()
        '''
        Args:
        
        f -- integer specifying the shape of window
        in_channels : number of channels of input to this identity block
        filters - list of integers (len=3 ) count of feature maps to be produced
        
        Returns 
        X -- output of the identity block,tensor of shape (n_H,n_W,n_C)
        '''
        self.in_channels = in_channels
        self.kernel_size = f
        self.F1 = filters[0]
        self.F2 = filters[1]
        self.F3 = filters[2]
        #first component 
        self.conv1 = self.conv_layer(in_channels=self.in_channels,out_channels=self.F1,kernel_size=(1,1),padding=0)
        self.bn1 = nn.BatchNorm2d(self.F1)
        #second component
        self.conv2 = self.conv_layer(in_channels=self.F1,out_channels=self.F2,kernel_size=(self.kernel_size,self.kernel_size),padding="same")
        self.bn2 = nn.BatchNorm2d(self.F2)
        #third component
        self.conv3 = self.conv_layer(in_channels=self.F2,out_channels=self.F3,kernel_size=(1,1),padding=0)
        self.bn3 = nn.BatchNorm2d(self.F3)

        
    def conv_layer(self,in_channels,out_channels,kernel_size,padding=0,stride=1):
        '''
        Args:
           in_channels : the number of input channel from the  input tensor
           out_channels : the number of output channels of the feature map 
           kernel_size  : filter size 
           padding      : that takes two values [same ,0]"[default : 0]
           stride       : the stride length [default is zero]

        Output:
            convolution layer
        '''
  
        if padding=="same":
            padding = int((kernel_size[0]-1)/2)
        return nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
    
    def forward(self,x):
        x_shortcut = x
      #  print("first",x_shortcut.shape,x.shape)
      #  x = self.conv_layer(in_channels=x.shape[1] , out_channels=self.F1,kernel_size=(1,1),padding=0)(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
       # print("after 1",x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        x =  F.relu(x)
        #print("after 2",x.shape)
        x = self.conv3(x)
        x = self.bn3(x)
        #print("after 3",x.shape)
        #print(x_shortcut.shape)
        x = x + x_shortcut
        del x_shortcut
        x = F.relu(x)
        
        return x
        

#convolutional block
class ConvolutionalBlock(nn.Module):
    '''
    Args:
    f           : the size of filter that will be used in the intermediate layers of this convolutional block
    in_channels : the number of input channels from the input tensor
    filters      :  list of integers (len=3 ) count of feature maps to be produced
    stride      : the length of stride
    
    
    '''
    def __init__(self,f,in_channels,filters,stride=2):
        super(ConvolutionalBlock,self).__init__()
        self.in_channels = in_channels
        self.kernel_size = f
        self.F1 = filters[0]
        self.F2 = filters[1]
        self.F3 = filters[2]
        self.stride = stride
        #first component
        self.conv1 = self.conv_layer(self.in_channels,self.F1,(1,1),padding=0,stride=self.stride)
        self.bn1 = nn.BatchNorm2d(self.F1)
        
        #second component
        self.conv2 = self.conv_layer(self.F1,self.F2,(f,f),padding="same")
        self.bn2 = nn.BatchNorm2d(self.F2)
        
        #third componenet
        self.conv3 = self.conv_layer(self.F2,self.F3,(1,1))
        self.bn3 = nn.BatchNorm2d(self.F3)
        
        #shortcut component
        self.sconv = self.conv_layer(self.in_channels,self.F3,(1,1),padding=0,stride=self.stride)
        self.sbn = nn.BatchNorm2d(self.F3)
        
    def forward(self,x):
        
        x_shortcut = x
#         print(x_shortcut.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)

        #
        #print(x_shortcut.shape)
        x_shortcut = self.sconv(x_shortcut)

        x_shortcut = self.sbn(x_shortcut)
        
        x = x+x_shortcut
        del x_shortcut
        x = F.relu(x)
        return x
        
        
    def conv_layer(self,in_channels,out_channels,kernel_size,padding=0,stride=1):
        '''
        Args:
           in_channels : the number of input channel from the  input tensor
           out_channels : the number of output channels of the feature map 
           kernel_size  : filter size 
           padding      : that takes two values [same ,0]"[default : 0]
           stride       : the stride length [default is zero]
        
        Output:
            Convolutional layer
        '''
        if padding=="same":
            padding = int((kernel_size[0]-1)/2)
        return nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,stride)
    
        
class ResNet50(nn.Module):
    
    def __init__(self,in_channels,identity_block,convolutional_block):
        super(ResNet50,self).__init__()
        
        self.in_channels = in_channels
        self.convolutional_block = convolutional_block
        self.identity_block = identity_block
        
        #stage1
        self.stage1_conv = self.conv_layer(self.in_channels,64,(7,7),padding=0,stride=2)
        self.stage1_bn = nn.BatchNorm2d(64)
        self.stage1_maxpool = nn.MaxPool2d(kernel_size=(3,3),stride=(2,2))
        
        #stage2
        self.stage2_convblock = self.convolutional_block(3,64,[64,64,256],1)
        self.stage2_identity_block1 = self.identity_block(3,256,[64,64,256])
        self.stage2_identity_block2 = self.identity_block(3,256,[64,64,256])
        
        #stage3
        self.stage3_convblock = self.convolutional_block(3,256,[128,128,512])
        self.stage3_identity_block1 = self.identity_block(3,512,[128,128,512])
        self.stage3_identity_block2 = self.identity_block(3,512,[128,128,512])
        self.stage3_identity_block3 = self.identity_block(3,512,[128,128,512])
        
        #stage 4
        self.stage4_convblock = self.convolutional_block(3,512,[128,128,1024])
        self.stage4_identity_block1 = self.identity_block(3,1024,[128,128,1024])
        self.stage4_identity_block2 = self.identity_block(3,1024,[128,128,1024])
        self.stage4_identity_block3 = self.identity_block(3,1024,[128,128,1024])
        self.stage4_identity_block4 = self.identity_block(3,1024,[128,128,1024])
        self.stage4_identity_block5 = self.identity_block(3,1024,[128,128,1024])
        
        #stage5
        self.stage5_convblock = self.convolutional_block(3,1024,[512,512,2048])
        self.stage5_identity_block1 = self.identity_block(3,2048,[512,512,2048])
        self.stage5_identity_block2 = self.identity_block(3,2048,[512,512,2048])
        #pooling
        self.average_pool = nn.AvgPool2d((2,2))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048,1)
        
    def forward(self,x):
        #stage1
        x = self.stage1_conv(x)
        x = self.stage1_bn(x)
        x = F.relu(x)
        x = self.stage1_maxpool(x)
        
        #stage2
        x = self.stage2_convblock(x)
        x = self.stage2_identity_block1(x)
        x = self.stage2_identity_block2(x)
        
        #stage3
        x = self.stage3_convblock(x)
        x = self.stage3_identity_block1(x)
        x = self.stage3_identity_block2(x)
        x = self.stage3_identity_block3(x)
        
        #stage4
        x = self.stage4_convblock(x)
        x = self.stage4_identity_block1(x)
        x = self.stage4_identity_block2(x)
        x = self.stage4_identity_block3(x)
        x = self.stage4_identity_block4(x)
        x = self.stage4_identity_block5(x)
        
        #stage5
        x = self.stage5_convblock(x)
        x = self.stage5_identity_block1(x)
        x = self.stage5_identity_block2(x)
        
        #pooling
        x = self.average_pool(x)
        x = self.gap(x)
        print(x.shape)
        x = x.squeeze()
        if len(x.shape)==1:
            print("single element")
            x = x.unsqueeze(0)

        #fully connected 
        x = self.fc(x)
        print(x.shape)
        
    def conv_layer(self,in_channels,out_channels,kernel_size,padding=0,stride=1):
        if padding=="same":
            padding = int((kernel_size[0]-1)/2)
        return nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)