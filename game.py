import numpy as np
import heuristics
import random
from dataclasses import dataclass

import torch
import torch.nn as nn

maxDepth = 3

@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray

class Game: 
    def __init__(self):
        self.board = np.zeros((6, 7), dtype = int)
        self.state = None  # 0 if drawn, {player} if won
        self.turn = 1    # Current player's turn
        self.column = -1 # Last picked column (starts at 0)
        self.history = []     # Contains all boards
        self.experiences = [] # Contains all experiences

    def game_move(self, picked_column, ouragent = False):

        # add new board to history, add new experience
        if ouragent:
            self.history.append(np.copy(self.board))
            if len(self.history) >= 2:
                self.experiences.append(Experience(self.history[-2], picked_column, None, self.history[-1]))

        if self.state != None:
            print("Game over")
            return

        # calls move from outside the class and throws error if column full
        if move(self.board, picked_column, self.turn) == -1:
            return -1

        # update last chosen column
        self.column = picked_column
        
        # check for win
        self.game_win()
        
        self.turn *= -1

        # nothing went wrong! returns 0
        return 0

    def game_win(self):
        if self.state == None:
            self.state = win(self.board, self.turn)

    def random_agent(self):
        return random.randint(0, 6)

    def minimax_agent(self, depth):
        return minimax(self.board, self.turn, 0, depth, -1, 1) # (eval, move)

    def q_agent(self, q_identity = None, player = 1):
        # create a new network, ask it for a move
        q = q_identity
        board = np.copy(self.board)
        # flatten board
        # board = np.reshape(board, (42))
        # TODO: reverse the board
        if player != 1:
            board = -1 * board
        #board = np.append(board, heur)
        q_choices = q.forward(torch.from_numpy(board).float())
        _, indices = torch.sort(q_choices) # ascending order
        return indices

    def playGame(self, agent1 = 1, agent2 = 1, pick_display = 0, epsilon = 0, minmaxrng = 0, first_player = 1):
        
        # keeps track of the q network's final move
        our_last_move = -1
        self.turn = 1 if first_player == 1 else -1
        
        # play game until either player wins or game draws
        while (self.state == None):
            # figure out who's turn it is
            ouragent = True if self.turn == 1 else False
            agent = agent1 if self.turn == 1 else agent2
            if agent == 1: 
                # random
                while self.game_move(self.random_agent(), ouragent) == -1:
                    pass
            elif agent == 3: 
                # minimax rng
                if random.random() < minmaxrng:
                    while self.game_move(self.minimax_agent(maxDepth)[1], ouragent) == -1:
                        pass
                else:
                    self.game_move(self.minimax_agent(maxDepth)[1], ouragent)
            else: 
                if random.random() < epsilon:
                    random_move = self.random_agent()
                    while self.game_move(random_move, ouragent) == -1:
                        random_move = self.random_agent()
                    # after our turn, update most recent move
                    if self.turn == -1:
                        our_last_move = random_move
                else:
                    # TODO: Modify this
                    q_choices = self.q_agent(agent, self.turn) # If q-agent, should NOT be 1 or 3
                    index_chosen = 6
                    while index_chosen >= 0 and self.game_move(q_choices[index_chosen], ouragent) == -1:
                        index_chosen -= 1
                    # after our turn, update most recent move
                    if self.turn == -1:
                        our_last_move = q_choices[index_chosen]
            # board display
            if pick_display == 1:
                display(self.board)

        self.experiences.append(Experience(self.history[-1], our_last_move, self.state, None))

    def get_computer_move(self, net):
        q_choices = self.q_agent(net, self.turn)
        index_chosen = 6
        while index_chosen >= 0 and self.game_move(q_choices[index_chosen], False) == -1:
            index_chosen -= 1

def display(board):
    output = ""
    for row in range(6):
        for column in range(7):
            if board[row][column] == 0:
                output += "."
            elif board[row][column] == 1:
                output += "X"
            elif board[row][column] == -1:
                output += "O"
            output += " "
        output += "\n"
    print(output)

def move(board, picked_column, turn):

    if picked_column == -1:
        return -1

    row = 5 
    while board[row][picked_column] != 0:
        row -= 1
        if row == -1:
            return -1
    
    board[row][picked_column] = turn

def win(board, turn):
    
    has_won = False

    dirs = [(0,1),(1,0),(1,1),(1,-1)] 

    for row in range(6):
        for col in range(7):
            for (dx, dy) in dirs:
                if 0 <= (row + 3 * dx) <= 5 and 0 <= (col + 3 * dy) <= 6:
                    # check 4 in a row in the direction of (dx, dy)
                    check_sum = 0
                    for i in range(4):
                        check_sum += board[row + i * dx][col + i * dy]
                    if check_sum == 4 * turn:
                        has_won = True
                
    # modify the state based on boolean
    if has_won:
        return turn
    else:
        empty_sum = 0
        for col in range(7):
            if board[0][col] == 0:
                empty_sum += 1
        if empty_sum == 0:
            return 0
    return None

def minimax(board, player, picked_column, depth, alpha, beta):
    # check if we win, lose, or draw
    won_player = win(board, player)
    if won_player == 1:
        return (2, picked_column)
    if won_player == -1:
        return (-2, picked_column)
    if won_player == 0:
        return (0, picked_column)
    # if depth is 0, then return heuristic value
    if depth == 0:
        return (heuristics.static_evaluation(board, player), -1)

    # if we are player 1, then:
        # value = -infinity
        # for each child of node, do:
            # set the value to max of (board, value, minimax(child, 2, depth - 1))
    # return value
    if player == 1:
        max_eval = float('-inf')
        best_move = -1
        for possible_move in range(7):
            mod_board = np.copy(board)
            if move(mod_board, possible_move, 1) == -1:
                continue
            val = minimax(mod_board, -1, possible_move, depth - 1, alpha, beta)[0]
            if val > max_eval:
                max_eval = val
                best_move = possible_move
            alpha = max(alpha, max_eval)
            if beta <= alpha:
                break
        return (max_eval, best_move)
    if player == -1:
        min_eval = float('+inf')
        best_move = -1
        for possible_move in range(7):
            mod_board = np.copy(board)
            if move(mod_board, possible_move, -1) == -1:
                continue
            val = minimax(mod_board, 1, possible_move, depth - 1, alpha, beta)[0]
            if val < min_eval:
                min_eval = val
                best_move = possible_move
            beta = min(beta, min_eval)
            if beta <= alpha:
                break
        return (min_eval, best_move)