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
    
    def __init__(self, x_data, y_data, a_data):
        
        """
        Store `x_data`. to `self.x`, `a_data` to `self.a`, and `y_data` to `self.y`.
        """ 
        self.x = x_data
        self.a = a_data
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
        a = self.a[index]
        y = self.y[index]
        # convert activity and labels to tensor
        a = torch.tensor(a, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, a, y


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
    file = '../../data/interim/PPG_FieldStudy_Windowed_Activity_Recognition/S1_labels.csv'
    ydf = pd.read_csv(file)
    #print(ydf.head(3))
    # merge data and labels
    data_subject = pd.merge(xdf, ydf, on="window_ID")
    # create custom dataset
    dataset = CustomDataset(data_subject['Data'], data_subject['predicted_activity'], data_subject['Label'])
    # split dataset to train and test data
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # load a batch of train dataset
    data_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    loader_iter = iter(data_loader)
    x, a, y = next(loader_iter)
    # check dimensions
    print("shapeof x:", x.shape)
    print("shapeof a:",a.shape)
    print("shapeof y:",y.shape)

    #print(x)
    #print(a)
    #print(y)

    # end time
    _END_RUNTIME = time.time()

    # total time taken
    print(f"Runtime is {_END_RUNTIME - _START_RUNTIME}")

    pass
  
    

if __name__ == '__main__':
    cli_main()
