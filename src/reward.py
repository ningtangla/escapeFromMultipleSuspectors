import numpy as np
# import env

class RewardFunctionTerminalPenalty():
    def __init__(self, sheepId, aliveBouns, deathPenalty, isTerminal):
        self.sheepId = sheepId
        self.aliveBouns = aliveBouns
        self.deathPenalty = deathPenalty
        self.isTerminal = isTerminal
    def __call__(self, state, action):
        reward = self.aliveBouns
        if self.isTerminal(state):
            reward = reward + self.deathPenalty
        return reward
