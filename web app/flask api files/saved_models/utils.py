
'''
LEANDER THUANI (LI)
'''
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
def preprocess_image(image_path):
    
    '''
    Args :
    image : the PIL type image
    Returns
    Preprocessed tensor 
    ''' 
    image = Image.open(image_path)
    
    preprocess = transforms.Compose([
    transforms.Resize((3,224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def regression_activation_mapping(image_tensor,feature,weights,image_shape):
    '''
    Args :
    image_tensor : image tensor with size of (1,3,224,224)
    feature     : feature tensor  with  size of (1,2048,7,7)
    weights     : weights with size of (2048,)
    Returns :
    Regression activation map with size of (3,224,224)
    '''
    
    size_upsample = (224, 224)
    
    bz, nc, h, w = feature.shape
    cam = np.dot(weights.reshape(1,-1),feature.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    up_sampled_image =cv2.resize(cam_img, size_upsample) 
    heatmap = cv2.applyColorMap(cv2.resize(up_sampled_image,image_shape), cv2.COLORMAP_JET)
    # return heatmap
    
    return heatmap
    # print(cam_img)