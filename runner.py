from chess_dataset import ChessDataset

def main():
    print("Hello world")
    train_dataset = ChessDataset()
    print(train_dataset.__getitem__(64))

if __name__ == '__main__':
    main()
