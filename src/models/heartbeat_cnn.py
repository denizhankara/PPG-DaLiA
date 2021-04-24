import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils. data import Dataloader


#load data into train and val
# need to chang
#train_dataset =
#val_dataset =

def load_data(train_dataset, val_dataset):
    train_loader = torch.utils.data.Dataloader(train_dataset, batch_size = 32, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 32, shuffle = False)

    return train_loader, val_loader

#initialize CNN model
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
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
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
        x = self.fc2

        return x

model = HeartbeatCNN()
print(model)

#initialize criterion and optimizer (need to fill with correct types)
criterion = torch.nn.modules.loss.CrossEntropyLoss
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

#set number of epochs
n_epochs = 5

def train_model(model, train_loader, n_epoch = n_epochs, optimizer=optimizer, criterion=criterion):

    model.train()  # prep model for training

    for epoch in range(n_epoch):
        curr_epoch_loss = []
        for data, target in train_loader:
            # your code here
            outputs = model(data)
            loss = criterion(outputs, target)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            curr_epoch_loss.append(loss.cpu().data.numpy())
        print(f"Epoch {epoch}: curr_epoch_loss={np.mean(curr_epoch_loss)}")
    return model


def eval_model(model, val_loader):
    model.eval()

    for data, target in val_loader:
        # your code here
        outputs = model(data)

