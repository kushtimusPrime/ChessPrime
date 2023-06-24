from chess_dataset import ChessDataset
from torch.utils.data import Dataset, DataLoader
from chess_net import module, ChessNet
from chess_loss import ChessLoss
import torch
import wandb
from torch.nn.parallel import DataParallel
import os
import torch.multiprocessing as mp

def main():
    cpu_count = mp.cpu_count()
    print("Number of CPU cores:", cpu_count)
    wandb.login()
    print("Starting training")
    total_game_limit_num = 12
    train_dataset = ChessDataset(total_game_limit = total_game_limit_num)
    print("Loaded train dataset")
    test_dataset = ChessDataset(total_game_limit=total_game_limit_num/4,is_train = False)
    print("Loaded test dataset")
    train_dataset_size = train_dataset.__len__()
    print("Train dataset size: " + str(train_dataset_size))
    batch_size = 32
    passes = train_dataset_size / batch_size
    train_dataloader = DataLoader(train_dataset,batch_size,shuffle=True,drop_last=True)
    test_dataloader = DataLoader(test_dataset,batch_size,shuffle=False)
    model = ChessNet()
    model_path = 'models/model.pth'
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully.")
    loss_fn = ChessLoss()
    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    num_epochs = int(train_dataset_size / batch_size)
    print("Num epochs: " + str(num_epochs))
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
            print("Progress: " + str(i) + "/ " + str(passes))

        if(i > 0):
            train_loss /= i
            training_losses.append(train_loss)
            print("Epoch " + str(epoch) + "/ " + str(num_epochs) +": " + str(train_loss))
            wandb.log({"Training Loss": train_loss})

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
                # Rolled up to the club like RDJ at a comic-con
                # Got so much green,looking like the son of Garmadon
        if(i > 0):
            test_loss /= i
            test_losses.append(test_loss)
            print("Test loss: " + str(test_loss))
            wandb.log({"Testing Loss": test_loss})
            if(test_loss <= min(test_losses)):
                if not os.path.exists("models"):
                    os.makedirs("models")
                torch.save(model.state_dict(),'models/model.pth')
                wandb.save('models/model.pth')
                print("Save model")

    print("Done")


if __name__ == '__main__':
    main()
