import numpy as np
# import env

class RewardFunctionTerminalPenaltyWithKillZoneOfProbability():
    def __init__(self, nonColisionReward, minDistance, calDistBetweenSheepAndSuspectors): #, actionCost, actionSpace):
        self.nonColisionReward = nonColisionReward
        self.minDistance = minDistance
        self.calDistBetweenSheepAndSuspectors = calDistBetweenSheepAndSuspectors
        #self.actionCost = actionCost
        #self.maxMagnitude = max([np.linalg.norm(action) for action in actionSpace])
    def __call__(self, state): #, action):
        physicalState, beliefAndAttention = state
        agentStates, agentActions, timeStep, wolfIdAndSubtlety = physicalState
        hypothesisInformation, positionOldTimeDF = beliefAndAttention

        distances = self.calDistBetweenSheepAndSuspectors(agentStates)
        isInKillZone = np.array(distances <= self.minDistance).astype(int)
        probsOfWolves = hypothesisInformation['identityProb'].groupby(['wolfIdentity']).mean().values

        reward = - np.sum(isInKillZone * probsOfWolves * 10)
        return reward

class RewardFunctionTerminalPenaltyWithKillZoneOfProbabilitySquare():
    def __init__(self, nonColisionReward, minDistance, calDistBetweenSheepAndSuspectors): #, actionCost, actionSpace):
        self.nonColisionReward = nonColisionReward
        self.minDistance = minDistance
        self.calDistBetweenSheepAndSuspectors = calDistBetweenSheepAndSuspectors
        #self.actionCost = actionCost
        #self.maxMagnitude = max([np.linalg.norm(action) for action in actionSpace])
    def __call__(self, state): #, action):
        physicalState, beliefAndAttention = state
        agentStates, agentActions, timeStep, wolfIdAndSubtlety = physicalState
        hypothesisInformation, positionOldTimeDF = beliefAndAttention

        distances = self.calDistBetweenSheepAndSuspectors(agentStates)
        isInKillZone = np.array(distances <= self.minDistance).astype(int)
        probsOfWolves = hypothesisInformation['identityProb'].groupby(['wolfIdentity']).mean().values

        reward = - np.sum(np.square(isInKillZone * probsOfWolves * 10))
        return reward
