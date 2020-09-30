import numpy as np
# import env

class RewardFunctionTerminalPenalty():
    def __init__(self, sheepId, aliveBouns, actionCost, deathPenalty, isTerminal, actionSpace):
        self.sheepId = sheepId
        self.aliveBouns = aliveBouns
        self.actionCost = actionCost
        self.deathPenalty = deathPenalty
        self.isTerminal = isTerminal
        self.maxMagnitude = max([np.linalg.norm(action) for action in actionSpace])
    def __call__(self, state, action):
        reward = self.aliveBouns
        if self.isTerminal(state):
            reward = reward + self.deathPenalty
        cost = self.actionCost * np.linalg.norm(action) / self.maxMagnitude
        reward = reward + cost
        return reward
