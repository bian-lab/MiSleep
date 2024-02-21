import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import os
from misleep import get_stanford_data_label
import numpy as np

torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True


def get_artifacts(EEG_data, EMG_data, threshold=5):
    """
    Identify artifacts' index with standard deviation, if the epoch's SD is threshold times
    than the average SD of this channel, identify it as artifact

    Args:
        EEG_data: 2D-array like, raw EEG epoch data
        EMG_data: 2D-array like, raw EMG epoch data
        threshold: int, control of artifact threshold

    Returns:
        A list of identified artifacts' index
    """

    EEG_SD_lst = [np.std(each) for each in EEG_data]
    EMG_SD_lst = [np.std(each) for each in EMG_data]
    ave_EEG_SD = np.mean(EEG_SD_lst)
    ave_EMG_SD = np.mean(EMG_SD_lst)
    artifacts_idx = []

    for idx, each in enumerate(EEG_SD_lst):
        if each / ave_EEG_SD >= threshold or EMG_SD_lst[idx] / ave_EMG_SD >= threshold:
            artifacts_idx.append(idx)

    return artifacts_idx


def z_score_self(lst):
    """
    Z socre list, the formula is x-mean / std

    Args:
        lst: list like, will be normalized

    Returns:
        Data list which has been z score normalized
    """

    lst_mean = np.mean(lst, axis=0)
    lst_std = np.std(lst, axis=0)
    normalized_data = (lst - lst_mean) / lst_std
    return normalized_data


class self_dataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = len(self.Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len


class CNN(nn.Module):
    def __init__(self, SR=256, epoch_length=5):
        super(CNN, self).__init__()

        # The first layer of convolution
        self.conv1 = nn.Sequential(
            # (2, SR*epoch_length) -> (16, SR*epoch_length - 5 + 1)
            nn.Conv1d(
                in_channels=2,
                out_channels=4,
                kernel_size=5,
            ),
            nn.ReLU(),
            # (16, SR*epoch_length - 5 + 1) -> (16, (SR*epoch_length - 5 + 1) / 2)
            nn.MaxPool1d(kernel_size=2)
        )
        # The second layer of revolution
        self.conv2 = nn.Sequential(
            # (16, SR*epoch_length - 5 + 1) -> (32, ((SR*epoch_length - 5 + 1) / 2) - 5 + 1)
            nn.Conv1d(4, 8, 5),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.out = nn.Linear(8 * int((int((SR * epoch_length - 5 + 1) / 2) - 5 + 1) / 2),
                             3)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        #         print(x.size(0), x.shape)
        x = x.view(x.size(0), x.shape[1] * x.shape[2])
        output = self.out(x)
        return output


# get data
X = np.load('E:/workplace/EEGProcessing/analysis_ipynb/data/X.npy')
Y = np.load('E:/workplace/EEGProcessing/analysis_ipynb/data/Y.npy')

idx = 100
plt.figure(figsize=(10, 3))
plt.plot(X[idx][0], color='blue', linewidth=0.5)
plt.show()

plt.figure(figsize=(10, 3))
plt.plot(X[idx][1], color='orange', linewidth=0.5)
plt.show()


# data loader
train_size = int(X.shape[0]*0.85)
test_size = X.shape[0] - train_size
dataset = self_dataset(X, Y)
del X
del Y
train_dataset, test_dataset = random_split(dataset,
                                           [train_size, test_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_x = torch.tensor(np.array([each[0] for each in test_dataset]))
test_y = torch.tensor(np.array([each[1] for each in test_dataset]))

# Model
cnn = CNN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
cnn.to(device)

# optimizer and loss function
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()


test_x = test_x.cuda()
test_x = test_x.to(torch.float32)
test_y = test_y.cuda()
test_y = test_y.to(torch.float32)

steplist = []
losslist = []
accuracylist = []

for epoch in range(2):
    for step, (x, y) in enumerate(train_loader):
        x = x.cuda()
        x = x.to(torch.float32)
        y = y.cuda()
        y = y.to(torch.float32)
        output = cnn(x)
        output = torch.max(output, 1)[1]
        loss = loss_func(output.float(), y.float())
        loss.requires_grad_()
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()

        if step % 1000 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].to("cpu").detach().numpy()
            accuracy = sum(pred_y == test_y.to("cpu").detach().numpy()) / test_y.size(0)
            print('Epoch:', epoch, ' | train loss:%.4f'%loss.item(), ' | test accuracy:%.2f'%accuracy)
            steplist.append(step)
            losslist.append(loss.item())
            accuracylist.append(accuracy)
