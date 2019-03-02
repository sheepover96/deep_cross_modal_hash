import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, input_size, hash_code_len, *args, **kwargs):
        super(CNNModel, self).__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=0)
        self.pool1 = nn.MaxPool2d(2) 
        self.relu = nn.ReLU(inplace=True) 
        self.lrn = nn.LocalResponseNorm(64)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=5, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2) 
        self.conv3 = nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.pool5 = nn.MaxPool2d(2) 
        self.conv6 = nn.Conv2d(256, 4096, kernel_size=1, stride=1, padding=0)
        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(4096, hash_code_len)

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
        x = self.conv5(x)
        x = self.pool5(x)

        batch_size = x.shape[0]
        #x = x.view(batch_size, -1)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        
        return out


class TextModel(nn.Module):
    def __init__(self, vocab_size, hash_code_len, *args, **kwargs):
        super(TextModel, self).__init__(*args, **kwargs)
        self.fc1 = nn.Linear(vocab_size, 8192)
        self.fc2 = nn.Linear(8192, hash_code_len)

    def forward(self, input):
        x = self.fc1(input)
        out = self.fc2(x)

        return out
