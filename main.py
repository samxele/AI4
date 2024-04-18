import numpy as np
from game import *
import heuristics

import importlib  
qnetwork = importlib.import_module("q-network")

q = qnetwork.DeepQNetworkConnect4()
q.train()

g = Game()
g.playGame(agent1 = 1, agent2 = q, pick_display = 1)