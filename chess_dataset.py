import torch
import os
from torch.utils.data import Dataset
import chess
import chess.pgn
import numpy as np

class ChessDataset(Dataset):
    def __init__(self,total_game_limit = float('inf'),is_train = True):
        self.data_path_ = ""
        if(is_train):
            self.data_path_ = "data/pgns/train"
        else:
            self.data_path_ = "data/pgns/test"
        self.pgn_file_names_ = os.listdir(self.data_path_)
        self.total_games_ = [];
        self.total_num_games_ = 0
        self.total_game_limit_ = total_game_limit
        self.letter_2_num_ = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e':4, 'f': 5, 'g': 6, 'h': 7}
        self.X_list_ = []
        self.y_list_ = []

        # Limits the training games to the total game limit amount
        for pgn_file_name in self.pgn_file_names_:
            pgn_data_path = self.data_path_ + "/" + pgn_file_name
            pgn = None
            pgn = open(pgn_data_path)
            thru_all_games = False
            while ((self.total_num_games_ < self.total_game_limit_) and (not thru_all_games)):
                try:
                    game = chess.pgn.read_game(pgn)
                    self.total_num_games_ = self.total_num_games_ + 1
                    if game is not None:
                        self.total_games_.append(game)
                    else:
                        thru_all_games = True
                except:
                    continue
                    #print("Tough")

        for game in range(len(self.total_games_)):
            board = chess.Board()
            for number, move in enumerate(self.total_games_[game].mainline_moves()):
                X = self.boardToRep(board)
                y = self.moveToRep(move,board)
                if(number % 2 == 1):
                    X *= -1
                self.X_list_.append(X.float())
                self.y_list_.append(y.float())

    def __len__(self):
        return len(self.X_list_)

    def __getitem__(self,index):
        return self.X_list_[index],self.y_list_[index]

    # board formatted as 
    # r n b q k b n r
    # p p p p p p p p
    # . . . . . . . .
    # . . . . . . . .
    # . . . . . . . .
    # . . . . . . . .
    # P P P P P P P P
    # R N B Q K B N R
    # Opponent is lowercase, and we are uppercase

    # board_rep is a 6x8x8 matrix where 6 refers to each type of piece, and 8x8 for the board
    # 1 is our piece and -1 is their piece
    def boardToRep(self,board):
        pieces = ['p','r','n','b','q','k']
        layers = []
        for piece in pieces:
            layers.append(self.createRepLayer(board,piece))
        board_rep = np.stack(layers)
        return torch.tensor(board_rep).unsqueeze(0)

    def createRepLayer(self,board,piece_char):
        board_str = str(board)
        board_str_list = board_str.split('\n')
        board_mat = []
        for i in range(len(board_str_list)):
            row_str = board_str_list[i]
            row_str_list = row_str.split(' ')
            for j in range(len(row_str_list)):
                if(row_str_list[j] == piece_char):
                    row_str_list[j] = -1
                elif(row_str_list[j] == piece_char.upper()):
                    row_str_list[j] = 1
                else:
                    row_str_list[j] = 0
            board_mat.append(row_str_list)
        return np.array(board_mat)

    def moveToRep(self,move,board):
        board.push(move)
        move_str = str(move)

        from_output_layer = np.zeros((8,8))
        from_row = 8 - int(move_str[1])
        from_column = self.letter_2_num_[move_str[0]]
        from_output_layer[from_row,from_column] = 1

        to_output_layer = np.zeros((8,8))
        to_row = 8 - int(move_str[3])
        to_column = self.letter_2_num_[move_str[2]]
        to_output_layer[to_row,to_column] = 1

        move_numpy_matrix = np.stack([from_output_layer,to_output_layer])
        return torch.tensor(move_numpy_matrix).unsqueeze(0)