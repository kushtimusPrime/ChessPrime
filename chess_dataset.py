import torch
import os
from torch.utils.data import Dataset
import chess
import chess.pgn
import numpy as np

class ChessDataset(Dataset):
    def __init__(self,total_game_limit = float('inf'),is_train = True):
        data_path = ""
        if(is_train):
            data_path = "data/pgns/train"
        else:
            data_path = "data/pgns/test"
        pgn_file_names = os.listdir(data_path)
        white_game_limit = total_game_limit / 2
        white_game_count = 0
        black_game_limit = total_game_limit / 2
        black_game_count = 0
        total_games = [];
        total_num_games = 0
        self.letter_2_num_ = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e':4, 'f': 5, 'g': 6, 'h': 7}
        self.X_list_ = []
        self.y_list_ = []
        elo_threshold = 2000
        print("Starting first loop",flush=True)
        file_count = 1
        temp = False
        # Limits the training games to the total game limit amount
        for pgn_file_name in pgn_file_names:
            pgn_data_path = data_path + "/" + pgn_file_name
            pgn = None
            pgn = open(pgn_data_path)
            thru_all_games = False
            
            while ((total_num_games < total_game_limit) and (not thru_all_games)):
                try:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        thru_all_games = True
                    else:
                        result = game.headers['Result']
                        # We only want winning games
                        if(result != '1/2-1/2'):
                            black_won = None
                            player_good = False
                            # White won
                            if(result == '1-0' and white_game_count < white_game_limit):
                                black_won = False
                                if('WhiteElo' in game.headers):
                                    white_elo = int(game.headers['WhiteElo'])
                                    if(white_elo > 2000):
                                        player_good = True

                            # Black won
                            elif(result == '0-1' and black_game_count < black_game_limit):
                                black_won = True
                                if('BlackElo' in game.headers):
                                    black_elo = int(game.headers['BlackElo'])
                                    if(black_elo > 2000):
                                        player_good = True
                            else:
                                print("White game count: " + str(white_game_count))
                                print("Black game count: " + str(black_game_count))
                                print(result)
                                if(white_game_count >= white_game_limit):
                                    print("No more white games")
                                elif(black_game_count >= black_game_limit):
                                    print("No more black games")
                                else:
                                    print("Weird result error")
                                    exit()
                            
                            try:
                                if(player_good):
                                    board = chess.Board()
                                    for number, move in enumerate(game.mainline_moves()):
                                        X = self.boardToRep(board)
                                        y = self.moveToRep(move,board)
                                        # Even numbers are white
                                        if(black_won and number % 2 == 1):
                                            X *= -1
                                            self.X_list_.append(X.float())
                                            self.y_list_.append(y.float())
                                        elif(not black_won and number % 2 == 0):
                                            self.X_list_.append(X.float())
                                            self.y_list_.append(y.float())
                                    
                                    total_num_games += 1
                                    if(black_won):
                                        black_game_count += 1
                                    else:
                                        white_game_count += 1
                                    print(total_num_games)
                            except AssertionError as e:
                                print(e)
                except UnicodeDecodeError as e:
                    print(e)
                                                     
        #         try:
        #             game = chess.pgn.read_game(pgn)
        #             if game is None:
        #                 thru_all_games = True
        #             else:
        #                 white_elo = int(game.headers['WhiteElo'])
        #                 black_elo = int(game.headers['BlackElo'])
        #                 if(white_elo >= elo_threshold and black_elo >= elo_threshold):
        #                     total_num_games = total_num_games + 1
        #                     if game is not None:
        #                         total_games.append(game)
        #                     else:
        #                         thru_all_games = True
        #                     print("Game num: " + str(total_num_games))
        #                 else:
        #                     print("Mid game")
        #         except Exception as e: 
        #             continue
        #     file_count += 1
        # print("Ending first loop",flush=True)
        # for game in range(len(total_games)):
        #     print("Game number: " + str(game),flush=True)
        #     board = chess.Board()
        #     for number, move in enumerate(total_games[game].mainline_moves()):
        #         try:
        #             X = self.boardToRep(board)
        #             y = self.moveToRep(move,board)
        #             if(number % 2 == 1):
        #                 X *= -1
        #             self.X_list_.append(X.float())
        #             self.y_list_.append(y.float())
        #         except:
        #             continue

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
        return torch.tensor(board_rep)

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
        return torch.tensor(move_numpy_matrix)
