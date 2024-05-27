import numpy as np
from game import *
import heuristics

import importlib  
#qnetwork = importlib.import_module("q-network")

class DeepQNetworkConnect4(nn.Module):
    # def __init__(self, env):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(42, 42),
            nn.ReLU(),
            nn.Linear(42, 20),
            nn.ReLU(),
            nn.Linear(20, 7),
        )

    def forward(self, x):
        return self.network(x)

q = DeepQNetworkConnect4()
r = DeepQNetworkConnect4()
#train()

g = Game()
g.playGame(agent1 = q, agent2 = r, pick_display = 1, first_player = 2)
for i in g.experiences:
    print(i)