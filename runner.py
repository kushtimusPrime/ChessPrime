from chess_dataset import ChessDataset
from torch.utils.data import Dataset, DataLoader
from chess_net import module, ChessNet

def main():
    print("Hello world")
    train_dataset = ChessDataset(total_game_limit = 10)
    batch_size = 32
    train_dataloader = DataLoader(train_dataset,batch_size,shuffle=True,drop_last=True)
    model = ChessNet()
    random_x_item = train_dataset.__getitem__(100)
    print(model.forward(random_x_item[0]).shape)
    print(random_x_item[1].shape)

if __name__ == '__main__':
    main()
