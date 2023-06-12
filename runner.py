from chess_dataset import ChessDataset
from torch.utils.data import Dataset, DataLoader
from chess_net import module, ChessNet
from chess_loss import ChessLoss
import torch

def main():
    train_dataset = ChessDataset(total_game_limit = 10)
    test_dataset = ChessDataset(total_game_limit=10)
    batch_size = 32
    train_dataloader = DataLoader(train_dataset,batch_size,shuffle=True,drop_last=True)
    test_dataloader = DataLoader(test_dataset,batch_size,shuffle=False)
    model = ChessNet()
    loss_fn = ChessLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        i = 0
        for inputs,labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs,labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            i = i + 1
        if(i > 0):
            train_loss /= i
            print("Epoch " + str(epoch) + ": " + str(train_loss))

        if(epoch % 10 == 0):
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

    print("Done")


if __name__ == '__main__':
    main()
