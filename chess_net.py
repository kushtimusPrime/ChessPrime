import torch
import torch.nn as nn
import torch.nn.functional as F

class module(nn.Module):

    def __init__(self,hidden_size):
        super(module, self).__init__()
        self.hidden_size_ = hidden_size
        self.conv1_ = nn.Conv2d(hidden_size,hidden_size,3,stride=1,padding=1)
        self.conv2_ = nn.Conv2d(hidden_size,hidden_size,3,stride=1,padding=1)
        self.bn1_ = nn.BatchNorm2d(hidden_size)
        self.bn2_ = nn.BatchNorm2d(hidden_size)
        self.activation1_ = nn.SELU()
        self.activation2_ = nn.SELU()

    def forward(self,x):
        
        x_input = torch.clone(x)
        x = self.conv1_(x)
        x = self.bn1_(x)
        x = self.activation1_(x)
        x = self.conv2_(x)
        x = self.bn2_(x)
        x = x + x_input
        x = self.activation2_(x)
        return x

class ChessNet(nn.Module):

    def __init__(self,hidden_layers=4,hidden_size=200):
        super(ChessNet,self).__init__()
        self.hidden_layers_ = hidden_layers
        self.input_layer_ = nn.Conv2d(6,hidden_size,3,stride=1,padding=1)
        self.module_list_ = nn.ModuleList([module(hidden_size) for i in range(hidden_layers)])
        self.output_layer_ = nn.Conv2d(hidden_size,2,3,stride=1,padding=1)

    def forward(self,x):
        x = self.input_layer_(x)
        x = F.relu(x)

        for i in range(self.hidden_layers_):
            x = self.module_list_[i](x)

        x = self.output_layer_(x)

        return x