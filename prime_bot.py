import chess
import chess.pgn
import numpy as np
import torch
from chess_net import module, ChessNet

def check_mate_single(board):
    board = board.copy()
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        board.push_uci(str(move))
        if board.is_checkmate():
            move = board.pop()
            return move
        _ = board.pop()

def distribution_over_moves(vals):
    probs = [tensor.detach() for tensor in vals]
    #probs = np.array(vals)
    probs = np.exp(probs)
    probs = probs / probs.sum()
    probs = probs ** 3
    probs = probs / probs.sum()
    return probs

def createRepLayer(board, piece_char):
    board_str = str(board)
    board_str_list = board_str.split("\n")
    board_mat = []
    for i in range(len(board_str_list)):
        row_str = board_str_list[i]
        row_str_list = row_str.split(" ")
        for j in range(len(row_str_list)):
            if row_str_list[j] == piece_char:
                row_str_list[j] = -1
            elif row_str_list[j] == piece_char.upper():
                row_str_list[j] = 1
            else:
                row_str_list[j] = 0
        board_mat.append(row_str_list)
    return np.array(board_mat)


def boardToRep(board):
    pieces = ["p", "r", "n", "b", "q", "k"]
    layers = []
    for piece in pieces:
        layers.append(createRepLayer(board, piece))
    board_rep = np.stack(layers)
    return torch.tensor(board_rep)


def main():
    letter_2_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e':4, 'f': 5, 'g': 6, 'h': 7}
    model = ChessNet()
    model_path = 'models/model.pth'
    model.load_state_dict(torch.load(model_path))
    print("Model loaded successfully.")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    board = chess.Board()
    my_move = True
    bot_color = ""
    print("Start chess bot. Are you (W)hite or (B)lack?")
    side = input()
    if side == "W":
        print("Bot is white")
        bot_color = "WHITE"
        my_move = True
    elif side == "B":
        print("Bot is black")
        bot_color = "BLACK"
        my_move = False
    else:
        print("Please select white or black next time")
        exit(0)

    while (
        (not board.is_checkmate())
        and (not board.is_stalemate())
        and (not board.is_insufficient_material())
    ):
        if my_move:
            print("Prime bot move")
            legal_moves = list(board.legal_moves)
            move = check_mate_single(board)
            if move is not None:
                board.push(move)
            else:
                x = torch.Tensor(boardToRep(board)).float().to(device)
                if(bot_color == "BLACK"):
                    x *= -1
                x = x.unsqueeze(0)
                move = model(x)
                the_real_move = legal_moves[0]
                move_val = -float('inf')
                move_vals = 0
                for legal_move in legal_moves:
                    from_move = str(legal_move)[:2]
                    from_val = move[0][0,:,:][8 - int(from_move[1]),letter_2_num[from_move[0]]].detach()
                    to_move = str(legal_move)[2:]
                    to_val = move[0][1,:,:][8 - int(to_move[1]),letter_2_num[to_move[0]]].detach()
                    total_val = from_val + to_val
                    move_vals = move_vals + total_val
                    if(total_val > move_val):
                        the_real_move = legal_move
                        move_val = total_val
                print(the_real_move)
                print("Confidence: " + str(move_val/move_vals))
                board.push(the_real_move)

                # vals = []
                # froms = [str(legal_move)[:2] for legal_move in legal_moves]
                # #print(froms)
                # froms = list(set(froms))
                # for from_ in froms:
                #     val = move[0][0,:,:][8 - int(from_[1]),letter_2_num[from_[0]]]
                #     vals.append(val)
                # probs = distribution_over_moves(vals)
                # choosen_from = str(np.random.choice(froms,size=1,p=probs)[0])[:2]

                # vals = []
                # for legal_move in legal_moves:
                #     from_ = str(legal_move)[:2]
                #     if from_ == choosen_from:
                #         to = str(legal_move)[2:]
                #         val = move[0][1,:,:][8 - int(to[1]),letter_2_num[to[0]]]
                #         val = val.detach()
                #         vals.append(val)
                #     else:
                #         vals.append(0)
                # choosen_move = legal_moves[np.argmax(vals)]
                # print(choosen_move)
                # board.push(choosen_move)
        else:
            print("Current Board")
            print("Capital is white")
            print("Lowercase is black\n")

            # Makes the board more readable
            readable_board = str(board)
            readable_board_list = readable_board.split("\n")
            current_number = 8
            for row in readable_board_list:
                row += " " + str(current_number)
                print(row)
                current_number -= 1
            print("a b c d e f g h")
            selected_move = False
            while not selected_move:
                print("Opponent move. Please enter in PGN format")
                opponent_move = input()
                try:
                    is_legal_move = (
                        chess.Move.from_uci(opponent_move) in board.legal_moves
                    )
                except:
                    print("Sorry pick a real move")
                if is_legal_move:
                    the_move = chess.Move.from_uci(opponent_move)
                    board.push(the_move)
                    print(board)
                    selected_move = True
                else:
                    print("Sorry pick a real move")
        my_move = not my_move


if __name__ == "__main__":
    main()
