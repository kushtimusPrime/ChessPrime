# Chess AI
Based on this link: https://towardsdatascience.com/creating-a-chess-ai-using-deep-learning-d5278ea7dcf

## Installation
Install necessary libraries with 

``` pip install -r requirements.txt ```

## Data
Got dataset from: https://www.kaggle.com/datasets/robikscube/this-week-in-chess-archive?resource=download
```
sudo apt-get update
sudo apt-get upgrade
cd ~/ChessPrime
sudo apt-get install unzip
pip3 install gdown
gdown 1_l05GZQ1ExDo-PuUOD0bKkU9QMra3iL7 -O ~/data.zip
unzip ~/data.zip
mkdir data
mv ~/pgns ~/ChessPrime/data
 ```

 We want to split test and train data 50/50.
 ```
 python3 split_data.py
 ```
 5 numbers will be printed. The first number is the total number of games in the dataset. The second and third number refers to the start and stop files that you should split it into. (For instance, if you get 1000 and 1339, your test folder should contain files twic1000.pgn to twic1339.pgn. All other files should go to test folder). The fourth number refers to the number of test games.
