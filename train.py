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
        self.board_ = chess.Board()
    
    def initializeData(self):
        for pgn_file_name in self.pgn_file_names_:
            print("Game count: " + str(self.total_num_games_))
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
        print("Game count: " + str(self.total_num_games_))

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


def main():
    chess_trainer = ChessTrainer(1000)
    chess_trainer.initializeData()
    matrix = chess_trainer.makeMatrix(chess_trainer.board_)
    print(chess_trainer.translate(matrix,chess_trainer.chess_dict_))

if __name__ == "__main__":
    main()
