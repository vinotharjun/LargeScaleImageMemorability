import torch
from config import *
import torch.nn as nn
from torchvision import models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    #loading the model
    model_50 = models.resnet50(pretrained=True)
    #defining the architecute
    for param in model_50.parameters():
        param.requires_grad = False
        n_inputs = model_50.fc.in_features
        last_layer = nn.Linear(n_inputs, 1)
        model_50.fc = last_layer
    model_50.to(device)
    loaded_data = torch.load(MODEL_WEIGHTS_PATH,map_location={"cuda:0":"cpu"})
    model_50.load_state_dict(loaded_data["model_state_dict"])
    model_50.eval()
    return model_50


