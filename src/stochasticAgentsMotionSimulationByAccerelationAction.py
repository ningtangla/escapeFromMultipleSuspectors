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
        startAgentPositions = np.array(self.resetAgentPositions(wolfId, distractorsIds))
        startAgentVelocities = np.array([[0, 0] for agentId in range(self.numAgent)])
        startTimeStep = np.array([0])
        startPhysicalState = [startAgentPositions, startAgentVelocities, startTimeStep, startWolfIdAndSubtlety]
        return startPhysicalState

class SheepPolicy():
    def __init__(self, updateFrequency, startMaxSheepSpeed, endMaxSheepSpeed, warmUpTimeSteps):
        self.updateFrequency = updateFrequency
        self.startMaxSheepSpeed = startMaxSheepSpeed
        self.endMaxSheepSpeed = endMaxSheepSpeed
        self.warmUpTimeSteps = warmUpTimeSteps
    def __call__(self, sheepPos, sheepAccer, oldSheepVel, timeStep):
        if timeStep % self.updateFrequency == 0:
            warmUpRate = min(1, timeStep/self.warmUpTimeSteps)
            sheepMaxSpeed = self.startMaxSheepSpeed + (self.endMaxSheepSpeed - self.startMaxSheepSpeed) * warmUpRate
            sheepVel = np.array(oldSheepVel) + np.array(sheepAccer)
            sheepSpeed = np.linalg.norm(sheepVel, ord = 2)
            if sheepSpeed > sheepMaxSpeed:
                sheepVel = np.array(sheepVel) / sheepSpeed * sheepMaxSpeed 
        else:
            sheepVel = np.array(oldSheepVel)
        return sheepVel

class WolfPolicy():
    def __init__(self, updateFrequency, minWolfSpeed, maxWolfSpeed, warmUpTimeSteps):
        self.updateFrequency = updateFrequency
        self.minWolfSpeed = minWolfSpeed
        self.maxWolfSpeed = maxWolfSpeed
        self.warmUpTimeSteps = warmUpTimeSteps
    def __call__(self, wolfPos, sheepPos, chasingSubtlety, oldWolfVel, timeStep):
        if timeStep % self.updateFrequency == 0:
            wolfPosition = np.array(wolfPos)      
            sheepPosition = np.array(sheepPos)
            heatSeekingDirectionPolar = ag.transiteCartesianToPolar(sheepPosition - wolfPosition)
            wolfDirectionPolar = np.random.vonmises(heatSeekingDirectionPolar, chasingSubtlety) 
            wolfDirection = ag.transitePolarToCartesian(wolfDirectionPolar)
            
            warmUpRate = min(1, timeStep/self.warmUpTimeSteps)
            wolfSpeed = self.minWolfSpeed + (self.maxWolfSpeed - self.minWolfSpeed) * warmUpRate
            wolfVel = wolfSpeed * wolfDirection
        else:
            wolfVel = np.array(oldWolfVel)
        return wolfVel

class DistractorPolicy():
    def __init__(self, updateFrequency, minDistractorSpeed, maxDistractorSpeed, warmUpTimeSteps):
        self.updateFrequency = updateFrequency
        self.minDistractorSpeed = minDistractorSpeed
        self.maxDistractorSpeed = maxDistractorSpeed
        self.warmUpTimeSteps = warmUpTimeSteps
    def __call__(self, distractorPos, oldDistractorVel, timeStep):
        if timeStep % self.updateFrequency == 0:
            distractorPosition = np.array(distractorPos)
            oldDistractorDirectionPolar = ag.transiteCartesianToPolar(oldDistractorVel)
            distractorDirectionPolar = np.random.uniform(-math.pi*1/3, math.pi*1/3) + oldDistractorDirectionPolar 
            distractorDirection = ag.transitePolarToCartesian(distractorDirectionPolar)
            
            warmUpRate = min(1, timeStep/self.warmUpTimeSteps)
            distractorSpeed = self.minDistractorSpeed + (self.maxDistractorSpeed - self.minDistractorSpeed) * warmUpRate
            distractorVel = distractorSpeed * distractorDirection
        else: 
            distractorVel = np.array(oldDistractorVel)
        return distractorVel

class PreparePolicy():
    def __init__(self, sheepId, numAgent, sheepPolicy, wolfPolicy, distractorPolicy):
        self.sheepId = sheepId
        self.agentIds = list(range(numAgent))
        self.sheepPolicy = sheepPolicy
        self.wolfPolicy = wolfPolicy
        self.distractorPolicy = distractorPolicy
    def __call__(self, agentPositions, oldAgentVelocities, timeStep, wolfId, wolfSubtlety, sheepOwnAction):
        oldSheepVel = oldAgentVelocities[self.sheepId]
        sheepPolicy = lambda sheepPos: self.sheepPolicy(sheepPos, sheepOwnAction, oldSheepVel, timeStep)
        oldWolfVel = oldAgentVelocities[wolfId]
        sheepPosForWolfPolicy = agentPositions[self.sheepId]
        wolfPolicy = lambda wolfPos : self.wolfPolicy(wolfPos, sheepPosForWolfPolicy, wolfSubtlety, oldWolfVel, timeStep)
        makeDistractorPolicy = lambda oldDistractorVel: lambda distractorPos: self.distractorPolicy(distractorPos, oldDistractorVel, timeStep)
        agentPolicyFunctions = [makeDistractorPolicy(oldDistractorVel) for oldDistractorVel in oldAgentVelocities]
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
        sheepAction = np.array(action)
        agentPolicyFunctions = self.preparePolicy(oldAgentPositions, oldAgentVelocites, timeStep, wolfId, wolfSubtlety, sheepAction) 
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

