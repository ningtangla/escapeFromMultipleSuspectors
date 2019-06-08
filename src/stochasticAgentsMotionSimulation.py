import numpy as np 
import AnalyticGeometryFunctions as ag
import math

class ResetAgentState():
    def __init__(self, initPosition, initPositionNoise):
        self.initPosition = initPosition
        self.initPositionNoiseLow, self.initPositionNoiseHigh = initPositionNoise
    def __call__(self):
        noiseDirectionPolar = np.random.uniform(0,360) /180 * math.pi
        noise = np.random.uniform(self.initPositionNoiseLow, self.initPositionNoiseHigh) * np.array([np.cos(noiseDirectionPolar), np.sin(noiseDirectionPolar)])
        startPosition = self.initPosition + np.array(noise)
        return startPosition

class ResetWolfIdAndSubtlety():
    def __init__(self, suspectorIds, possibleSubtleties):
        self.suspectorIds = suspectorIds
        self.possibleSubtleties = possibleSubtleties
    def __call__(self):
        wolfId = np.random.choice(self.suspectorIds)
        subtlety = np.random.choice(self.possibleSubtleties)
        startWolfIdAndSubtlety = [wolfId, subtlety]
        return startWolfIdAndSubtlety

class ResetPhysicalState():
    def __init__(self, sheepId, numAgent, resetSheepState, resetWolfOrDistractorState, resetWolfIdAndSubtlety):
        self.sheepId = sheepId
        self.agentIds = list(range(numAgent))
        self.resetSheepState = resetSheepState
        self.resetWolfOrDistractorState = resetWolfOrDistractorState
        self.resetWolfIdAndSubtlety = resetWolfIdAndSubtlety
    def __call__(self):   
        resetAgentStateFunctions = [self.resetWolfOrDistractorState for agentId in self.agentIds]
        resetAgentStateFunctions[self.sheepId] = self.resetSheepState
        startAgentStates = np.array([resetAgentStateFunctions[agentId]() for agentId in self.agentIds])
        startAgentActions = np.array([[None, None] for agentId in self.agentIds])
        startTimeStep = np.array([0])
        startWolfIdAndSubtlety = self.resetWolfIdAndSubtlety()
        startPhysicalState = [startAgentStates, startAgentActions, startTimeStep, startWolfIdAndSubtlety]
        return startPhysicalState

class SheepPolicy():
    def __init__(self, updateFrequency, minSheepSpeed, maxSheepSpeed, warmUpTimeSteps):
        self.updateFrequency = updateFrequency
        self.minSheepSpeed = minSheepSpeed
        self.maxSheepSpeed = maxSheepSpeed
        self.warmUpTimeSteps = warmUpTimeSteps
    def __call__(self, sheepState, sheepAction, oldSheepAction, timeStep):
        if timeStep % self.updateFrequency == 0:
            warmUpRate = min(1, timeStep/self.warmUpTimeSteps)
            sheepSpeed = self.minSheepSpeed + (self.maxSheepSpeed - self.minSheepSpeed) * warmUpRate
            sheepAction = np.array(sheepAction) * sheepSpeed 
        else:
            sheepAction = np.array(oldSheepAction)
        return sheepAction

class WolfPolicy():
    def __init__(self, updateFrequency, minWolfSpeed, maxWolfSpeed, warmUpTimeSteps):
        self.updateFrequency = updateFrequency
        self.minWolfSpeed = minWolfSpeed
        self.maxWolfSpeed = maxWolfSpeed
        self.warmUpTimeSteps = warmUpTimeSteps
    def __call__(self, wolfState, sheepState, chasingSubtlety, oldWolfAction, timeStep):
        if timeStep % self.updateFrequency == 0:
            wolfPosition = np.array(wolfState)      
            sheepPosition = np.array(sheepState)
            heatSeekingDirectionPolar = ag.transiteCartesianToPolar(sheepPosition - wolfPosition)
            wolfDirectionPolar = np.random.vonmises(heatSeekingDirectionPolar, chasingSubtlety) 
            wolfDirection = ag.transitePolarToCartesian(wolfDirectionPolar)
            
            warmUpRate = min(1, timeStep/self.warmUpTimeSteps)
            wolfSpeed = self.minWolfSpeed + (self.maxWolfSpeed - self.minWolfSpeed) * warmUpRate
            wolfAction = wolfSpeed * wolfDirection
        else:
            wolfAction = np.array(oldWolfAction)
        return wolfAction

class DistractorPolicy():
    def __init__(self, updateFrequency, minDistractorSpeed, maxDistractorSpeed, warmUpTimeSteps):
        self.updateFrequency = updateFrequency
        self.minDistractorSpeed = minDistractorSpeed
        self.maxDistractorSpeed = maxDistractorSpeed
        self.warmUpTimeSteps = warmUpTimeSteps
    def __call__(self, distractorState, oldDistractorAction, timeStep):
        if timeStep % self.updateFrequency == 0:
            distractorPosition = np.array(distractorState)      
            distractorDirectionPolar = np.random.uniform(-math.pi, math.pi) 
            distractorDirection = ag.transitePolarToCartesian(distractorDirectionPolar)
            
            warmUpRate = min(1, timeStep/self.warmUpTimeSteps)
            distractorSpeed = self.minDistractorSpeed + (self.maxDistractorSpeed - self.minDistractorSpeed) * warmUpRate
            distractorAction = distractorSpeed * distractorDirection
        else: 
            distractorAction = np.array(oldDistractorAction)
        return distractorAction

class PreparePolicy():
    def __init__(self, sheepId, numAgent, sheepPolicy, wolfPolicy, distractorPolicy):
        self.sheepId = sheepId
        self.agentIds = list(range(numAgent))
        self.sheepPolicy = sheepPolicy
        self.wolfPolicy = wolfPolicy
        self.distractorPolicy = distractorPolicy
    def __call__(self, agentStates, oldAgentActions, timeStep, wolfId, wolfSubtlety, action):
        oldSheepAction = oldAgentActions[self.sheepId]
        sheepPolicy = lambda sheepState: self.sheepPolicy(sheepState, action, oldSheepAction, timeStep)
        oldWolfAction = oldAgentActions[wolfId]
        sheepStateForWolfPolicy = agentStates[self.sheepId]
        wolfPolicy = lambda wolfState : self.wolfPolicy(wolfState, sheepStateForWolfPolicy, wolfSubtlety, oldWolfAction, timeStep)

        agentPolicyFunctions = [lambda distractorState: self.distractorPolicy(distractorState, oldDistractiorAction, timeStep) for
                oldDistractiorAction in oldAgentActions]
        agentPolicyFunctions[self.sheepId] = sheepPolicy
        agentPolicyFunctions[wolfId] = wolfPolicy
        return agentPolicyFunctions


class TransiteMultiAgentMotion():
    def __init__(self, checkBoundaryAndAdjust):
        self.checkBoundaryAndAdjust = checkBoundaryAndAdjust
    def __call__(self, agentPositions, agentVelocities):
        newAgentPositions = np.array(agentPositions) + np.array(agentVelocities)
        checkedNewAgentPositionsAndVelocities = [self.checkBoundaryAndAdjust(position, velocity) for position, velocity in zip(newAgentPositions, agentVelocities)]
        newAgentPositions, newAgentVelocities = list(zip(*checkedNewAgentPositionsAndVelocities))

        return newAgentPositions, newAgentVelocities


class UpdatePhysicalState():
    def __init__(self, numAgent, preparePolicy):
        self.agentIds = list(range(numAgent))
        self.preparePolicy = preparePolicy
    def __call__(self, oldPhysicalState, action):  
        oldAgentStates, oldAgentActions, timeStep, wolfIdAndSubtlety = oldPhysicalState
        wolfId, wolfSubtlety = wolfIdAndSubtlety
        sheepOwnAction = np.array(action)
        agentPolicyFunctions = self.preparePolicy(oldAgentStates, oldAgentActions, timeStep, wolfId, wolfSubtlety, sheepOwnAction) 
        agentActions = [agentPolicyFunctions[agentId](oldAgentStates[agentId]) for agentId in self.agentIds]
        timeStep = timeStep + 1
        updatedAgentActionsPhysicalState = [oldAgentStates, agentActions, timeStep, wolfIdAndSubtlety]
        return updatedAgentActionsPhysicalState


class CheckBoundaryAndAdjust():
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary
    def __call__(self, position, velocity):
        adjustedX, adjustedY = position
        adjustedVelX, adjustedVelY = velocity
        if position[0] >= self.xMax:
            adjustedX = 2 * self.xMax - position[0]
            adjustedVelX = -velocity[0]
        if position[0] <= self.xMin:
            adjustedX = 2 * self.xMin - position[0]
            adjustedVelX = -velocity[0]
        if position[1] >= self.yMax:
            adjustedY = 2 * self.yMax - position[1]
            adjustedVelY = -velocity[1]
        if position[1] <= self.yMin:
            adjustedY = 2 * self.yMin - position[1]
            adjustedVelY = -velocity[1]
        checkedPosition = np.array([adjustedX, adjustedY])
        checkedVelocity = np.array([adjustedVelX, adjustedVelY])
        return checkedPosition, checkedVelocity

