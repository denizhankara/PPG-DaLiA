#!/usr/bin/env python
# coding: utf-8
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    
    def __init__(self, x_data, y_data):
        
        """
        Store `x_data`. to `self.x` and `y_data` to `self.y`.
        """ 
        self.x = x_data
        self.y = y_data
            
    def __len__(self):
        
        """
        Return the number of samples (i.e. windows)
        """
        return len(self.x)
        
    
    def __getitem__(self, index):
        
        """
        Generates one sample of data.
        """
        if torch.is_tensor(index):
            index = index.tolist()
        
        x = self.x[index]
        y = self.y[index]
        # convert labels to tensor
        y = torch.tensor(y, dtype=torch.float32)

        return x, y




class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=4,out_channels=8,kernel_size=(1,1),stride=(1,1))
        # self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=(1,1))
        # self.maxpool2 = nn.MaxPool2d(kernel_size=(1,2),stride=(1,2))
        # self.conv3 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(1,3),stride=(1,1)) 
        # self.maxpool3 = nn.MaxPool2d(kernel_size=(1,2),stride=(1,2))
        # self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(1,3),stride=(1,1))
        # self.fc1 = nn.Linear(1*126*32,1*126*32)
        # self.fc2 = nn.Linear(512,2)
        pass
        
    def forward(self, x):
        #input is of shape (batch_size=32, 3, 1025, 4)
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x,1,2)
        # x = F.relu(self.conv2(x))  
        # x = F.max_pool2d(x,2,2)
        
        # x = x.view(-1,13*13*4)        
           
        # x = F.relu(self.fc1(x))        
        # x = self.fc2(x)
        # return(x)
        pass


def cli_main():
    _START_RUNTIME = time.time()
    # set seed
    seed = 96710
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # read pickle for subject 1
    x_pkl = pd.read_pickle("../../data/interim/PPG_FieldStudy_CNN_Input/S1.pkl")
    # create dataframe
    xdf = pd.DataFrame(list(x_pkl.items()),columns = ['window_ID','Data'])
    # read lables for subject 1
    file = '../../data/interim/PPG_FieldStudy_Windowed/S1_labels.csv'
    ydf = pd.read_csv(file)
    # merge data and labels
    data_subject = pd.merge(xdf, ydf, on="window_ID")
    # create custom dataset
    dataset = CustomDataset(data_subject['Data'], data_subject['Label'])
    # split dataset to train and test data
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # load a batch of train dataset
    data_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    loader_iter = iter(data_loader)
    x, y = next(loader_iter)
    # check dimensions
    print(x.shape)
    print(y.shape)

    pass
  
    

if __name__ == '__main__':
    cli_main()
