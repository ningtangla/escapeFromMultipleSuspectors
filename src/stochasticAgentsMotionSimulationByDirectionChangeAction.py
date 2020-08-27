import numpy as np 
import AnalyticGeometryFunctions as ag
import math

class ResetWolfIdAndSubtlety():
    def __init__(self, suspectorIds, possibleSubtleties):
        self.suspectorIds = suspectorIds
        self.possibleSubtleties = possibleSubtleties
    def __call__(self):
        wolfId = np.random.choice(self.suspectorIds)
        subtlety = np.random.choice(self.possibleSubtleties)
        startWolfIdAndSubtlety = [wolfId, subtlety]
        return startWolfIdAndSubtlety

class IsLegalInitPositions():
    def __init__(self, sheepId, minSheepWolfDistance, minSheepDistractorDistance):
        self.sheepId = sheepId
        self.minSheepWolfDistance = minSheepWolfDistance
        self.minSheepDistractorDistance = minSheepDistractorDistance

    def __call__(self, initPositions, wolfId, distractorsIds):
        sheepPosition = initPositions[self.sheepId]
        wolfPosition = initPositions[wolfId]
        distractorsPositions = [initPositions[id] for id in distractorsIds]
        sheepWolfDistance = np.linalg.norm((np.array(sheepPosition) - np.array(wolfPosition)), ord=2)
        sheepDistractorsDistances = [np.linalg.norm((np.array(sheepPosition) - np.array(distractorPosition)), ord=2) 
                for distractorPosition in distractorsPositions]
        legalSheepWolf = (sheepWolfDistance > self.minSheepWolfDistance)
        legalSheepDistractors = np.all([(sheepDistractorDistance > self.minSheepDistractorDistance) for sheepDistractorDistance in sheepDistractorsDistances])  
        legal = legalSheepWolf and legalSheepDistractors
        return legal

class ResetAgentPositions():
    def __init__(self, xBoundary, yBoundary, numOfAgent, isLegalInitPositions):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary
        self.numOfAgnet = numOfAgent
        self.isLegalInitPositions = isLegalInitPositions

    def __call__(self, wolfId, distractorsIds):
        initAllAgentsPositions = [[np.random.uniform(self.xMin, self.xMax),
                      np.random.uniform(self.yMin, self.yMax)]
                     for _ in range(self.numOfAgnet)]
        while not self.isLegalInitPositions(initAllAgentsPositions, wolfId, distractorsIds):
            initAllAgentsPositions = [[np.random.uniform(self.xMin, self.xMax),
                          np.random.uniform(self.yMin, self.yMax)]
                         for _ in range(self.numOfAgnet)] 
        
        initPositions = np.array(initAllAgentsPositions)
        return initPositions

class ResetPhysicalState():
    def __init__(self, sheepId, numAgent, resetAgentPositions, resetWolfIdAndSubtlety):
        self.sheepId = sheepId
        self.numAgent = numAgent
        self.resetAgentPositions = resetAgentPositions
        self.resetWolfIdAndSubtlety = resetWolfIdAndSubtlety
    def __call__(self):   
        startWolfIdAndSubtlety = self.resetWolfIdAndSubtlety()
        wolfId, subtlety = startWolfIdAndSubtlety
        distractorsIds = [id for id in range(self.numAgent) if id not in [self.sheepId, wolfId]]  
        startAgentStates = np.array(self.resetAgentPositions(wolfId, distractorsIds))
        startAgentActions = np.array([ag.transitePolarToCartesian(np.random.uniform(-math.pi, math.pi)) for agentId in range(self.numAgent)])
        startTimeStep = np.array([0])
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
            oldDistractorDirectionPolar = ag.transiteCartesianToPolar(oldDistractorAction)
            distractorDirectionPolar = np.random.uniform(-math.pi*1/3, math.pi*1/3) + oldDistractorDirectionPolar 
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
    def __call__(self, agentStates, oldAgentActions, timeStep, wolfId, wolfSubtlety, sheepOwnAction):
        oldSheepAction = oldAgentActions[self.sheepId]
        sheepPolicy = lambda sheepState: self.sheepPolicy(sheepState, sheepOwnAction, oldSheepAction, timeStep)
        oldWolfAction = oldAgentActions[wolfId]
        sheepStateForWolfPolicy = agentStates[self.sheepId]
        wolfPolicy = lambda wolfState : self.wolfPolicy(wolfState, sheepStateForWolfPolicy, wolfSubtlety, oldWolfAction, timeStep)
        makeDistractorPolicy = lambda oldDistractorAction: lambda distractorState: self.distractorPolicy(distractorState, oldDistractorAction, timeStep)
        agentPolicyFunctions = [makeDistractorPolicy(oldDistractiorAction) for oldDistractiorAction in oldAgentActions]
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
    def __init__(self, sheepId, numAgent, preparePolicy):
        self.sheepId = sheepId
        self.agentIds = list(range(numAgent))
        self.preparePolicy = preparePolicy
    def __call__(self, oldPhysicalState, action):  
        oldAgentPositions, oldAgentVelocites, timeStep, wolfIdAndSubtlety = oldPhysicalState
        wolfId, wolfSubtlety = wolfIdAndSubtlety
        normalizedSheepAction = np.array(action)/np.linalg.norm(action, ord = 2)
        sheepOldVelocity = oldAgentVelocites[self.sheepId]
        normalizedSheepOldVelocity  = np.array(sheepOldVelocity)/np.linalg.norm(sheepOldVelocity, ord = 2)
        sheepOwnVelocity = normalizedSheepOldVelocity + normalizedSheepAction 
        normalizedSheepOwnVelocity = sheepOwnVelocity/np.linalg.norm(sheepOwnVelocity, ord = 2)
        agentPolicyFunctions = self.preparePolicy(oldAgentPositions, oldAgentVelocites, timeStep, wolfId, wolfSubtlety, normalizedSheepOwnVelocity) 
        agentActions = [agentPolicyFunctions[agentId](oldAgentPositions[agentId]) for agentId in self.agentIds]
        agentVelocities = agentActions.copy()
        timeStep = timeStep + 1
        updatedAgentActionsPhysicalState = [oldAgentPositions, agentVelocities, timeStep, wolfIdAndSubtlety]
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

