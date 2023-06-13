from chess_dataset import ChessDataset
from torch.utils.data import Dataset, DataLoader
from chess_net import module, ChessNet
from chess_loss import ChessLoss
import torch

# This file just verifies that we can successfully load models back up
def main():
    batch_size = 32
    test_dataset = ChessDataset(total_game_limit=10)
    print("Loaded test dataset")
    test_dataloader = DataLoader(test_dataset,batch_size,shuffle=False)
    model = ChessNet()
    model.load_state_dict(torch.load('models/model.pth'))
    loss_fn = ChessLoss()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    test_loss = 0
    i = 0
    with torch.no_grad():
        for inputs,labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs,labels)
            test_loss += loss.item()
            i = i + 1
    if(i > 0):
        test_loss /= i
        print("Test loss: " + str(test_loss))

if __name__ == '__main__':
    main()