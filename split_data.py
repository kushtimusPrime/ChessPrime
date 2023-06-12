import csv

total_games = 0
with open('data/metadata.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    i = 1
    for row in spamreader:
        row_list = row[0].split(',')
        if(i > 1):
            total_games += int(row_list[1])
        i = i + 1
print(total_games)
split_game = total_games / 2
test_game_start = None
test_game_stop = None

game_count = 0
with open('data/metadata.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    i = 1
    for row in spamreader:
        row_list = row[0].split(',')
        if(i > 1):
            game_count += int(row_list[1])
            if(i == 2):
                test_game_start = row_list[0]
            if(game_count > split_game):
                test_game_stop = row_list[0]
                print(test_game_start)
                print(test_game_stop)
                print(game_count)
                exit()
        i = i + 1
