import os
from pathlib import Path, PurePath
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
from tqdm import tqdm
import re
from sklearn.metrics import mean_absolute_error

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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



def load_data(train_dataset, val_dataset):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 32, shuffle = False)

    return train_loader, val_loader



# CNN model
class HeartbeatCNN(nn.Module):
    def __init__(self):
        super(HeartbeatCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, (1, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, (3, 3), stride=(1, 1))
        self.conv3 = nn.Conv2d(16, 32, (1, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(32, 64, (1, 3), stride=(1, 1))
        self.conv5 = nn.Conv2d(64, 128, (1, 3), stride=(1, 1))
        self.conv6 = nn.Conv2d(128, 256, (1, 3), stride=(1, 1))
        self.conv7 = nn.Conv2d(256, 512, (1, 3), stride=(1, 1))
        self.conv8 = nn.Conv2d(512, 1024, (1, 3), stride=(1, 1))
        self.conv9 = nn.Conv2d(1024, 2048, (1, 3), stride=(1, 1))
        self.conv10 = nn.Conv2d(2048, 32, (1, 1), stride=(1, 1))
        self.pool = nn.MaxPool2d((1, 2), (1, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(192, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        x = self.pool(F.relu(self.conv7(x)))
        x = self.pool(F.relu(self.conv8(x)))
        x = self.pool(F.relu(self.conv9(x)))
        x = F.relu(self.conv10(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x



def train_model(model, train_loader, n_epoch = None, optimizer=None, criterion=None):

    model.train()  # prep model for training

    for epoch in range(n_epoch):
        curr_epoch_loss = []
        for data, target in tqdm(train_loader, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):

            data, target = data.to(device), target.to(device)
            outputs = model.forward(data)
            outputs = outputs.view(outputs.size(0))

            loss = criterion(outputs, target)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            #curr_epoch_loss.append(loss.cpu().data.numpy())
            curr_epoch_loss.append(loss.cpu().detach().numpy())
        print(f"Epoch {epoch}: curr_epoch_loss={np.mean(curr_epoch_loss)}")
    return model



def eval_model(model, val_loader):
    model.eval()

    Y_pred = []
    Y_true = []

    for data, target in tqdm(val_loader, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        
        data, target = data.to(device), target.to(device)
        
        # run the inputs through the model
        outputs = model.forward(data)

        # get predicted and target values
        pred_value = outputs.detach().cpu().numpy()
        target_value = target.detach().cpu().numpy()

        # append values to the lists
        Y_pred.append(pred_value)
        Y_true.append(target_value)

    # concatenate predictions and test data
    Y_pred = np.concatenate(Y_pred, axis=0)
    Y_true = np.concatenate(Y_true, axis=0)

    return Y_pred, Y_true


def cli_main():
    # set device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Get all the processed data in interim folder
    data_path = "../../data/interim/PPG_FieldStudy_CNN_Input/"
    # find all files in folder
    files = [f.path for f in os.scandir(data_path) if f.is_file()]
    # sort in order by subject (numerical part)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    # Get all the labels data in interim folder
    labels_path = "../../data/interim/PPG_FieldStudy_Windowed/"
    labels = [f.path for f in os.scandir(labels_path) if f.is_file() and 'labels' in PurePath(f).name]
    # sort in order by subject (numerical part)
    labels.sort(key=lambda f: int(re.sub('\D', '', f)))
    #print(files)
    #print(labels)

    #initialize CNN model
    model = HeartbeatCNN()
    print(model)
    model.to(device)

    #initialize criterion and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    #set number of epochs
    n_epochs = 1

    # iterate over all subjects
    for f, l in zip(files, labels):
        # current subject to process
        current_subject = os.path.splitext(os.path.basename(f))[0]
        
        # read pickle for subject
        x_pkl = pd.read_pickle(f)

        # create dataframe
        xdf = pd.DataFrame(list(x_pkl.items()),columns = ['window_ID','Data'])

        # read lables for subject
        ydf = pd.read_csv(l)

        # merge data and labels
        data_subject = pd.merge(xdf, ydf, on="window_ID")

        # create custom dataset
        dataset = CustomDataset(data_subject['Data'], data_subject['Label'])

        # split dataset to train and test data
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        #load data into train and val
        train_loader, val_loader = load_data(train_dataset, val_dataset)

        # train model
        model = train_model(model, train_loader, n_epoch = n_epochs, optimizer=optimizer, criterion=criterion)

        # evaluate model
        y_pred, y_true = eval_model(model, val_loader)

        # calculate model performance
        mae = mean_absolute_error(y_true, y_pred)
        print(f"Subject {current_subject}: mae={mae}")

if __name__ == '__main__':
    cli_main()
