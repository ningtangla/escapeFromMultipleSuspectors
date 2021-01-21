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
            physicalState, beliefAndAttention = state
            agentStates, agentActions, timeStep, wolfIdAndSubtlety = physicalState

            hypothesisInformation, positionOldTimeDF = beliefAndAttention
            posteriorOnIdentity = hypothesisInformation['identitySoftenProb'].groupby(['wolfIdentity']).mean().values
            wolfProb = posteriorOnIdentity[wolfIdAndSubtlety[0] - 1]
            #print(wolfProb)
            reward = reward + self.deathPenalty * min(1.0, 1.0 * wolfProb)

        cost = self.actionCost * 1.0 * np.linalg.norm(action) / self.maxMagnitude
        reward = reward - cost
        return reward
