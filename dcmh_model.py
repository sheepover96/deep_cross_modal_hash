import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, input_size, hash_code_len, *args, **kwargs):
        super(CNNModel, self).__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(input_size, 64, kernel_size=11, stride=4, padding=0)
        self.pool1 = nn.MaxPool2d(2) 
        self.relu = nn.ReLU() 
        self.lrn = nn.LocalResponseNorm(64)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=5, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2) 
        self.conv3 = nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=1)
        self.fc1 = nn.Linear(256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, hash_code_len)

    def forward(self, input):
        x = self.conv1(input)
        x = self.pool1(x)
        x = self.lrn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.lrn(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)

        out = self.fc3(x)
        
        return out


class TextModel(nn.Module):
    def __init__(self, vocab_size, hash_code_len, *args, **kwargs):
        super(BWModel, self).__init__(*args, **kwargs)
        self.fc1 = nn.Linear(vocab_size, 8192)
        self.fc2 = nn.Linear(8192, hash_code_len)

    def forward(self, input):
        x = self.fc1(input)
        out = self.fc2(x)

        return out