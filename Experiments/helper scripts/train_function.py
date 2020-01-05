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

def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()
    running_loss_history = []
    val_running_loss_history=[]
    orignal_model=None
    best_model_wts = copy.deepcopy(model.state_dict())
    low_loss = np.inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batched_data in tqdm(dataloaders[phase]):
          
                inputs=batched_data["image"]
                inputs = inputs.to(device)
                labels=batched_data["memorability_score"]
                labels=labels.view(-1,1).double()
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    print("  batch loss:    ",loss.item())
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase=="train":
              running_loss_history.append(epoch_loss)
            else:
              val_running_loss_history.append(epoch_loss)
            

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            

            # deep copy the model
            if phase == 'val' and epoch_loss < low_loss:
                low_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                print("saving best model....")
                torch.save(best_model_wts,"/content/drive/My Drive/image memorability/saved models/ResNet50_15-12-19.pth")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    original_model =copy.deepcopy(model)
    model.load_state_dict(best_model_wts)
    return model,original_model,running_loss_history,val_running_loss_history