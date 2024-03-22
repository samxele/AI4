import numpy as np
from game import *
import heuristics
'''
DRAWN POSITION:

thegame = game.Game()
thegame.display()
"""a = [
["OXOXOXO"]
["OXOXOXO"]
["XOXOXOX"]
["XOXOXOX"]
["OXOXOXO"]
["XOXOXOX"]
]"""
#a = "OOXXOX"[::-1]
# X ROW
for i in range(7):
    thegame.game_move(i)
    # O ROW
for i in range(7):
    thegame.game_move(i)
    # X ROW
for i in range(7):
    thegame.game_move(i)
thegame.game_move(1) # O
thegame.game_move(0) # X
thegame.game_move(3) # O 
thegame.game_move(2) # X 
thegame.game_move(5) # O 
thegame.game_move(4) # X

thegame.game_move(1) # O
thegame.game_move(0) # X
thegame.game_move(3) # O 
thegame.game_move(2) # X 
thegame.game_move(5) # O 
thegame.game_move(4) # X

thegame.game_move(0) # O
thegame.game_move(6) # X
thegame.game_move(6) # O

for i in range(1,7):
    thegame.game_move(i)

display(thegame)

'''
special = Game()

special.playGame(pick_display = 1)