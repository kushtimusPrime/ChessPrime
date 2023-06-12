import torch
import torch.nn as nn

class ChessLoss(nn.Module):

    def __init__(self):
        super(ChessLoss,self).__init__()

    def forward(self,prediction,target):
        from_loss_function = nn.CrossEntropyLoss()
        to_loss_function = nn.CrossEntropyLoss()
        from_loss = from_loss_function(prediction[:,0,:],target[:,0,:])
        to_loss = to_loss_function(prediction[:,1,:],target[:,1,:])
        total_loss = from_loss + to_loss
        return total_loss

