from chess_dataset import ChessDataset
from torch.utils.data import Dataset, DataLoader
from chess_net import module, ChessNet
from chess_loss import ChessLoss
import torch
import wandb

def main():
    wandb.login()
    train_dataset = ChessDataset(total_game_limit = 10)
    print("Loaded train dataset")
    test_dataset = ChessDataset(total_game_limit=10)
    print("Loaded test dataset")
    batch_size = 32
    train_dataloader = DataLoader(train_dataset,batch_size,shuffle=True,drop_last=True)
    test_dataloader = DataLoader(test_dataset,batch_size,shuffle=False)
    model = ChessNet()
    loss_fn = ChessLoss()
    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    num_epochs = 20
    training_losses = []
    test_losses = []
    run = wandb.init(
        # Set the project where this run will be logged
        project="chess_net_1",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
        })
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
            training_losses.append(train_loss)
            print("Epoch " + str(epoch) + ": " + str(train_loss))
            wandb.log({"Training Loss": train_loss})

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
                test_losses.append(test_loss)
                print("Test loss: " + str(test_loss))
                wandb.log({"Testing Loss": test_loss})
                if(test_loss <= min(test_losses)):
                    torch.save(model.state_dict(),'models/model.pth')
                    wandb.save('models/model.pth')
                    print("Save model")

    print("Done")


if __name__ == '__main__':
    main()
