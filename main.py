import numpy as np
from game import *
import heuristics

import importlib  
#qnetwork = importlib.import_module("q-network")

#q = qnetwork.DeepQNetworkConnect4()
#train()

g = Game()
g.playGame(agent1 = 3, agent2 = 3, pick_display = 1)