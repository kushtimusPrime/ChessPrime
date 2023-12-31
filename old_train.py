import os
import chess
import chess.pgn
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

class ChessTrainer:
    def __init__(self,total_game_limit = 100000):
        self.data_path_ = "data/pgns"
        self.pgn_file_names_ = os.listdir(self.data_path_)
        self.total_games_ = [];
        self.total_num_games_ = 0
        self.total_game_limit_ = total_game_limit
        self.letter_2_num_ = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e':4, 'f': 5, 'g': 6, 'h': 7}
        self.num_2_letter_ = {0: 'a', 1: 'b', 2:'c', 3:'d', 4:'e', 5: 'f', 6: 'g', 7:'h'}
        self.chess_dict_ = {
            'p' : [1,0,0,0,0,0,0,0,0,0,0,0],
            'P' : [0,0,0,0,0,0,1,0,0,0,0,0],
            'n' : [0,1,0,0,0,0,0,0,0,0,0,0],
            'N' : [0,0,0,0,0,0,0,1,0,0,0,0],
            'b' : [0,0,1,0,0,0,0,0,0,0,0,0],
            'B' : [0,0,0,0,0,0,0,0,1,0,0,0],
            'r' : [0,0,0,1,0,0,0,0,0,0,0,0],
            'R' : [0,0,0,0,0,0,0,0,0,1,0,0],
            'q' : [0,0,0,0,1,0,0,0,0,0,0,0],
            'Q' : [0,0,0,0,0,0,0,0,0,0,1,0],
            'k' : [0,0,0,0,0,1,0,0,0,0,0,0],
            'K' : [0,0,0,0,0,0,0,0,0,0,0,1],
            '.' : [0,0,0,0,0,0,0,0,0,0,0,0],
        }
        self.alpha_dict_ = {
            'a' : [0,0,0,0,0,0,0],
            'b' : [1,0,0,0,0,0,0],
            'c' : [0,1,0,0,0,0,0],
            'd' : [0,0,1,0,0,0,0],
            'e' : [0,0,0,1,0,0,0],
            'f' : [0,0,0,0,1,0,0],
            'g' : [0,0,0,0,0,1,0],
            'h' : [0,0,0,0,0,0,1],
        }
        self.number_dict_ = {
            1 : [0,0,0,0,0,0,0],
            2 : [1,0,0,0,0,0,0],
            3 : [0,1,0,0,0,0,0],
            4 : [0,0,1,0,0,0,0],
            5 : [0,0,0,1,0,0,0],
            6 : [0,0,0,0,1,0,0],
            7 : [0,0,0,0,0,1,0],
            8 : [0,0,0,0,0,0,1],
        }
        self.X_ = []
        self.y_ = []
    
    def initializeData(self):
        for pgn_file_name in self.pgn_file_names_:
            #print("Game count: " + str(self.total_num_games_))
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
                    print("Tough")
        #print("Game count: " + str(self.total_num_games_))

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
        return board_rep

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
        print(board)
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

        return np.stack([from_output_layer,to_output_layer])

    def makeMatrix(self,board): 
        pgn = board.epd()
        foo = []  
        pieces = pgn.split(" ", 1)[0]
        rows = pieces.split("/")
        for row in rows:
            foo2 = []  
            for thing in row:
                if thing.isdigit():
                    for i in range(0, int(thing)):
                        foo2.append('.')
                else:
                    foo2.append(thing)
            foo.append(foo2)
        return foo

    def translate(self,matrix,chess_dict):
        rows = []
        for row in matrix:
            terms = []
            for term in row:
                terms.append(chess_dict[term])
            rows.append(terms)
        return rows

    def dataSetup(self):
        print("Start data setup")
        for game in range(len(self.total_games_)):
            print("Game: " + str(game))
            board = chess.Board()
            for number, move in enumerate(self.total_games_[game].mainline_moves()):
                self.moveToRep(move,board)
                print("Move Number: " + str(number))
                #print("Move: " + str(move))
                #board.push(move)
                #move_str = str(move)
                #from_output_layer = np.zeros((8,8))
                #from_row = 8 - int(move_str[1])
                #from_column = self.letter_2_num_[move_str[0]]
                #from_output_layer[from_row,from_column] = 1


                #print(from_output_layer)
                #if(number > 1):
                #    exit()
            
        print("End data setup")


def main():
    chess_trainer = ChessTrainer(1)
    chess_trainer.initializeData()
    chess_trainer.dataSetup()
    #chess_trainer.dataSetup()

if __name__ == "__main__":
    main()
