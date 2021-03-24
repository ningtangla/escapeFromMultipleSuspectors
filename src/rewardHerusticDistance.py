import numpy as np
import pandas as pd
# import env

class ValueByLogDistance():
    def __init__(self, nonColisionReward, minDistance, rangeToCareSuspectors, calDistBetweenSheepAndSuspectors): #, actionCost, actionSpace):
        self.nonColisionReward = nonColisionReward
        self.minDistance = minDistance
        self.calDistBetweenSheepAndSuspectors = calDistBetweenSheepAndSuspectors
        self.rangeToCareSuspectors = rangeToCareSuspectors
        #self.actionCost = actionCost
        #self.maxMagnitude = max([np.linalg.norm(action) for action in actionSpace])
    def __call__(self, state): #, action):
        physicalState, beliefAndAttention = state
        agentStates, agentActions, timeStep, wolfIdAndSubtlety = physicalState
        hypothesisInformation, positionOldTimeDF = beliefAndAttention

        distances = self.calDistBetweenSheepAndSuspectors(agentStates)
        isOutKillZone = np.array(distances > self.minDistance).astype(int)
        distanceValueImpactByProb = hypothesisInformation['identityProb'].groupby(['wolfIdentity']).mean().values
        isWorthToAvoid =  np.array(hypothesisInformation['distanceBetweenWolfAndSheep'].groupby(['wolfIdentity']).mean().values <= self.rangeToCareSuspectors).astype(int)

        reward = np.sum(isOutKillZone * np.log(distances * 1.0 / self.minDistance) * distanceValueImpactByProb * isWorthToAvoid)
        #print(reward)
        #print(isOutKillZone, distances, distanceValueImpactByProb, isWorthToAvoid)
        #cost = self.actionCost * 1.0 * np.linalg.norm(action) / self.maxMagnitude
        #reward = reward - cost
        return reward


class ValueBySqrtDistance():
    def __init__(self, nonColisionReward, minDistance, rangeToCareSuspectors, calDistBetweenSheepAndSuspectors): #, actionCost, actionSpace):
        self.nonColisionReward = nonColisionReward
        self.minDistance = minDistance
        self.calDistBetweenSheepAndSuspectors = calDistBetweenSheepAndSuspectors
        self.rangeToCareSuspectors = rangeToCareSuspectors
        #self.actionCost = actionCost
        #self.maxMagnitude = max([np.linalg.norm(action) for action in actionSpace])
    def __call__(self, state): #, action):
        physicalState, beliefAndAttention = state
        agentStates, agentActions, timeStep, wolfIdAndSubtlety = physicalState
        hypothesisInformation, positionOldTimeDF = beliefAndAttention

        distances = self.calDistBetweenSheepAndSuspectors(agentStates)
        isOutKillZone = np.array(distances <= self.minDistance).astype(int)
        distanceValueImpactByProb = hypothesisInformation['identityProb'].groupby(['wolfIdentity']).mean().values
        isWorthToAvoid =  np.array(hypothesisInformation['distanceBetweenWolfAndSheep'].groupby(['wolfIdentity']).mean().values <= self.rangeToCareSuspectors).astype(int)

        reward = np.sum(isOutKillZone * np.sqrt(distances * 1.0 / self.minDistance) * distanceValueImpactByProb * isWorthToAvoid)
        #cost = self.actionCost * 1.0 * np.linalg.norm(action) / self.maxMagnitude
        #reward = reward - cost
        return reward
