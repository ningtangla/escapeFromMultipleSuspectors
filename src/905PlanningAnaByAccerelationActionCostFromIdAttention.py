import numpy as np
import pygame as pg
import pandas as pd
import pylab as plt
import itertools as it
import math
import os
import pathos.multiprocessing as mp
import copy

from anytree import AnyNode as Node
from anytree import RenderTree
from collections import OrderedDict

# Local import
from algorithms.stochasticPW import ScoreChild, SelectAction, SelectNextState, InitializeChildren, Expand, ExpandNextState, \
        PWidening, RollOut, backup, OutputAction, PWMultipleTrees

from simple1DEnv import TransitionFunction, RewardFunction, Terminal
from visualize import draw
import stochasticAgentsMotionSimulationByAccerelationActionBurnTimeWithDamp as ag
import Attention
import calPosterior as calPosterior
import stochasticBeliefAndAttentionSimulationBurnTimeUpdateIdentitySampleAttention as ba
import env
import rewardWithActionCostAndWolfProbability as reward
import trajectoriesSaveLoad as tsl
import AnalyticGeometryFunctions as agf

class MakeDiffSimulationRoot():
    def __init__(self, isTerminal, updatePhysicalStateByBelief, reUpdatePhysicalStateByBelief):
        self.isTerminal = isTerminal
        self.updatePhysicalStateByBelief = updatePhysicalStateByBelief
        self.reUpdatePhysicalStateByBelief = reUpdatePhysicalStateByBelief
    def __call__(self, action, state):
        stateForSimulationRoot = self.updatePhysicalStateByBelief(state)
        if self.isTerminal(stateForSimulationRoot):
            for resample in range(100):
                stateForSimulationRoot = self.reUpdatePhysicalStateByBelief(state)
                self.reUpdatePhysicalStateByBelief.softParaForIdentity*= 0.1
                if not self.isTerminal(stateForSimulationRoot):
                    self.reUpdatePhysicalStateByBelief.softParaForIdentity = 1
                    break
        rootNode = Node(id={action: stateForSimulationRoot}, numVisited=0, sumValue=0, is_expanded = False)
        return rootNode

class RunMCTSTrjactory:
    def __init__(self, maxRunningSteps, numTree, planFrequency, actionUpdateFrequecy, transitionFunctionInPlay, isTerminal, makeDiffSimulationRoot, render):
        self.maxRunningSteps = maxRunningSteps
        self.numTree = numTree
        self.planFrequency = planFrequency
        self.actionUpdateFrequecy = actionUpdateFrequecy
        self.transitionFunctionInPlay = transitionFunctionInPlay
        self.isTerminal = isTerminal
        self.makeDiffSimulationRoot = makeDiffSimulationRoot
        self.render = render

    def __call__(self, mcts):
        state, action = None, None
        currState = self.transitionFunctionInPlay(state, action)
        currStateOnTruth = copy.deepcopy(currState)
        oppoActionCurrStateOnTruth = copy.deepcopy(currState)
        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(currState):
                trajectory.append([currState, None, None, None, None, None])
                break
            if runningStep % (self.planFrequency * self.actionUpdateFrequecy) == 0:
                rootNodes = [self.makeDiffSimulationRoot(action, currState) for treeIndex in range(self.numTree)]
                actions = mcts(rootNodes)
                rootNodesOnTruth = [Node(id={action: currState}, numVisited=0, sumValue=0, is_expanded = False) for _ in range(self.numTree)]
                actionsOnTruth = np.array([[0, 3.4]]) #mcts(rootNodesOnTruth)
                #actionsOnTruth = np.array([0, 0])
                #actionSpace = [(np.cos(degreeInPolar), np.sin(degreeInPolar)) for degreeInPolar in np.arange(0, 360, 8)/180 * math.pi] 
                #actions = [actionSpace[np.random.choice(range(len(actionSpace)))]]
            action = actions[int(runningStep/self.actionUpdateFrequecy) % self.planFrequency]
            actionOnTruth = actionsOnTruth[int(runningStep/self.actionUpdateFrequecy) % self.planFrequency]
            oppoActionOnTruth = -1 * np.array(actionOnTruth)
            
            physicalState, beliefAndAttention = currState
            hypothesisInformation, positionOldTimeDF = beliefAndAttention
            probabilityOnHypothesisAttention = np.exp(hypothesisInformation['logP']) 
            posteriorOnHypothesisAttention = probabilityOnHypothesisAttention/probabilityOnHypothesisAttention.sum()
            probabilityOnAttentionSlotByGroupbySum = posteriorOnHypothesisAttention.groupby(['wolfIdentity','sheepIdentity']).sum().values
            posterior = probabilityOnAttentionSlotByGroupbySum/np.sum(probabilityOnAttentionSlotByGroupbySum)
            stateToRecord = [physicalState, posterior]
            
            physicalStateOnTruth, beliefAndAttentionOnTruth = currStateOnTruth
            oppoActionPhysicalStateOnTruth, beliefAndAttentionOnTruthOppoAction = oppoActionCurrStateOnTruth
            
            trajectory.append([stateToRecord, action, physicalStateOnTruth, oppoActionPhysicalStateOnTruth, 
                actionOnTruth, [np.array(list(rootNode.id.values())[0])[0][3] for rootNode in rootNodes]]) 
            #print(trajectory[-1][0][0][3], trajectory[-1][0][0][2])
            #print(trajectory[-1][5])
            currStateOnTruth = copy.deepcopy(currState)
            oppoActionCurrStateOnTruth = copy.deepcopy(currState)

            oppoActionNextStateOnTruth = self.transitionFunctionInPlay(oppoActionCurrStateOnTruth, oppoActionOnTruth)
            oppoActionCurrStateOnTruth = oppoActionNextStateOnTruth
            nextStateOnTruth = self.transitionFunctionInPlay(currStateOnTruth, actionOnTruth)
            currStateOnTruth = nextStateOnTruth
            #print(currStateOnTruth[1])
            nextState = self.transitionFunctionInPlay(currState, action)
            currState = nextState
            #print(trajectory[-1][5], self.transitionFunctionInPlay.updateBeliefAndAttention.attention.memoryratePerSlot,
            #        self.transitionFunctionInPlay.updatePhysicalStateByBelief.softParaForSubtlety)
            #print(currState[1])
            #print('***', currState[0][3], trajectory[-1][5])
            #print('***', physicalState[0][0], action)
            #print('***', np.linalg.norm(physicalState[1][0]), np.linalg.norm(physicalState[1][10]))
        return trajectory

class RunOneCondition:
    def __init__(self, getTrajectorySavePathByCondition, getCSVSavePathByCondition):
        self.getTrajectorySavePathByCondition = getTrajectorySavePathByCondition
        self.getCSVSavePathByCondition = getCSVSavePathByCondition
    
    def __call__(self, condition):
        
        getSavePath = self.getTrajectorySavePathByCondition(condition)
        attentionType = condition['attType']
        alpha = condition['alpha']
        C = condition['C']
        minAttentionDistance = condition['minAttDist']
        rangeAttention = condition['rangeAtt']
        numTree = condition['numTrees']
        numSimulations = condition['numSim']
        actionRatio = condition['actRatio']
        cBase = condition['cBase']
        burnTime = condition['burnTime']
        softParaForIdentity = condition['softId']
        softParaForSubtlety = condition['softSubtlety']
        damp = condition['damp']
        actionCost = condition['actCost']

        numSub = 10
        allIdentityResults = []
        allPerceptionResults = []
        allActionResults = []
        allVelDiffResults = []
        allResults = []
        possibleTrialSubtleties = [0.92, 0.01]#[500.0, 3.3, 1.83, 0.92, 0.01]
        for subIndex in range(numSub):
            meanIdentiyOnConditions = {}
            meanPerceptionOnConditions = {}
            meanActionOnConditions = {}
            meanVelDiffOnConditions = {}
            meanEscapeOnConditions = {}
            for chasingSubtlety in possibleTrialSubtleties: 

                print(numTree, chasingSubtlety, numSimulations, attentionType)
                numAgent = 25
                sheepId = 0
                suspectorIds = list(range(1, numAgent))

                resetWolfIdAndSubtlety = ag.ResetWolfIdAndSubtlety(suspectorIds, [chasingSubtlety])
                distanceToVisualDegreeRatio = 20
                minInitSheepWolfDistance = 9 * distanceToVisualDegreeRatio
                minInitSheepDistractorDistance = 2.5 * distanceToVisualDegreeRatio  # no distractor in killzone when init
                isLegalInitPositions = ag.IsLegalInitPositions(sheepId, minInitSheepWolfDistance, minInitSheepDistractorDistance)
                xBoundary = [0, 640]
                yBoundary = [0, 480]
                resetAgentPositions = ag.ResetAgentPositions(xBoundary, yBoundary, numAgent, isLegalInitPositions)
                resetPhysicalState = ag.ResetPhysicalState(sheepId, numAgent, resetAgentPositions, resetWolfIdAndSubtlety) 
               
                numFramePerSecond = 20
                numMDPTimeStepPerSecond = 5 
                numFrameWithoutActionChange = int(numFramePerSecond/numMDPTimeStepPerSecond)
                
                sheepActionUpdateFrequency = 1 
                minSheepSpeed = int(17.4 * distanceToVisualDegreeRatio/numFramePerSecond)
                maxSheepSpeed = int(23.2 * distanceToVisualDegreeRatio/numFramePerSecond)
                warmUpTimeSteps = int(10 * numMDPTimeStepPerSecond)
                sheepPolicy = ag.SheepPolicy(sheepActionUpdateFrequency, minSheepSpeed, maxSheepSpeed, warmUpTimeSteps, burnTime, damp)
                
                wolfActionUpdateFrequency = int(0.2 * numMDPTimeStepPerSecond)
                minWolfSpeed = int(8.7 * distanceToVisualDegreeRatio/numFramePerSecond)
                maxWolfSpeed = int(14.5 * distanceToVisualDegreeRatio/numFramePerSecond)
                wolfPolicy = ag.WolfPolicy(wolfActionUpdateFrequency, minWolfSpeed, maxWolfSpeed, warmUpTimeSteps)
                distractorActionUpdateFrequency = int(0.2 * numMDPTimeStepPerSecond)
                minDistractorSpeed = int(8.7 * distanceToVisualDegreeRatio/numFramePerSecond)
                maxDistractorSpeed = int(14.5 * distanceToVisualDegreeRatio/numFramePerSecond)
                distractorPolicy = ag.DistractorPolicy(distractorActionUpdateFrequency, minDistractorSpeed, maxDistractorSpeed, warmUpTimeSteps)
                preparePolicy = ag.PreparePolicy(sheepId, numAgent, sheepPolicy, wolfPolicy, distractorPolicy)
                updatePhysicalState = ag.UpdatePhysicalState(sheepId, numAgent, preparePolicy)

                xBoundary = [0, 640]
                yBoundary = [0, 480]
                checkBoundaryAndAdjust = ag.CheckBoundaryAndAdjust(xBoundary, yBoundary) 
                transiteMultiAgentMotion = ag.TransiteMultiAgentMotion(checkBoundaryAndAdjust)
               
                minDistance = 2.5 * distanceToVisualDegreeRatio
                isTerminal = env.IsTerminal(sheepId, minDistance)
               # screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
               # screenColor = np.array([0, 0, 0])
               # sheepColor = np.array([0, 255, 0])
               # wolfColor = np.array([255, 0, 0])
               # circleSize = 10
               # saveImage = True
               # saveImageFile = 'image3'
               # render = env.Render(numAgent, screen, xBoundary[1], yBoundary[1], screenColor, sheepColor, wolfColor, circleSize, saveImage, saveImageFile, isTerminal)
                render = None
                renderOnInSimulation = False
                transiteStateWithoutActionChangeInSimulation = env.TransiteStateWithoutActionChange(numFrameWithoutActionChange, isTerminal, transiteMultiAgentMotion, render, renderOnInSimulation)     
                renderOnInPlay = False
                transiteStateWithoutActionChangeInPlay = env.TransiteStateWithoutActionChange(numFrameWithoutActionChange, isTerminal, transiteMultiAgentMotion, render, renderOnInPlay)     
               
                if attentionType == 'idealObserver':
                    attentionLimitation= 1
                    precisionPerSlot=500.0
                    precisionForUntracked=500.0
                    memoryratePerSlot=1.0
                    memoryrateForUntracked=1.0
                if attentionType == 'preAttention':
                    attentionLimitation= 1
                    precisionPerSlot=2.5
                    precisionForUntracked=2.5
                    memoryratePerSlot=0.45
                    memoryrateForUntracked=0.45
                if attentionType == 'attention3':
                    attentionLimitation= 3
                    precisionPerSlot=8.0
                    precisionForUntracked=0.01
                    memoryratePerSlot=0.7
                    memoryrateForUntracked=0.01
                if attentionType == 'hybrid3':
                    attentionLimitation= 3
                    precisionPerSlot=8.0
                    precisionForUntracked=2.5
                    memoryratePerSlot=0.7
                    memoryrateForUntracked=0.45
                if attentionType == 'attention4':
                    attentionLimitation= 4
                    precisionPerSlot=8.0
                    precisionForUntracked=0.01
                    memoryratePerSlot=0.7
                    memoryrateForUntracked=0.01
                if attentionType == 'hybrid4':
                    attentionLimitation= 4
                    precisionPerSlot=8.0
                    precisionForUntracked=2.5
                    memoryratePerSlot=0.7
                    memoryrateForUntracked=0.45
                
                
                if attentionType == 'preAttentionMem0.25':
                    attentionLimitation= 1
                    precisionPerSlot=2.5
                    precisionForUntracked=2.5
                    memoryratePerSlot=0.25
                    memoryrateForUntracked=0.25
                if attentionType == 'preAttentionMem0.65':
                    attentionLimitation= 1
                    precisionPerSlot=2.5
                    precisionForUntracked=2.5
                    memoryratePerSlot=0.65
                    memoryrateForUntracked=0.65
                if attentionType == 'preAttentionPre0.5':
                    attentionLimitation= 1
                    precisionPerSlot=0.5
                    precisionForUntracked=0.5
                    memoryratePerSlot=0.45
                    memoryrateForUntracked=0.45
                if attentionType == 'preAttentionPre4.5':
                    attentionLimitation= 1
                    precisionPerSlot=4.5
                    precisionForUntracked=4.5
                    memoryratePerSlot=0.45
                    memoryrateForUntracked=0.45

                attention = Attention.AttentionToPrecisionAndDecay(precisionPerSlot, precisionForUntracked, memoryratePerSlot, memoryrateForUntracked)    
                transferMultiAgentStatesToPositionDF = ba.TransferMultiAgentStatesToPositionDF(numAgent)
                possibleSubtleties = [500.0, 11.0, 3.3, 1.83, 0.92, 0.31, 0.01]
                resetBeliefAndAttention = ba.ResetBeliefAndAttention(sheepId, suspectorIds, possibleSubtleties, attentionLimitation, transferMultiAgentStatesToPositionDF, attention)
               
                maxAttentionDistance = minAttentionDistance + rangeAttention
                attentionMinDistance = minAttentionDistance * distanceToVisualDegreeRatio
                attentionMaxDistance = maxAttentionDistance * distanceToVisualDegreeRatio
                numStandardErrorInDistanceRange = 4
                calDistancePriorOnAttentionSlot = Attention.CalDistancePriorOnAttentionSlot(attentionMinDistance, attentionMaxDistance, numStandardErrorInDistanceRange)
                attentionSwitch = Attention.AttentionSwitch(attentionLimitation, calDistancePriorOnAttentionSlot)    
                computePosterior = calPosterior.CalPosteriorLog(minDistance)

                print(attentionLimitation, attentionMinDistance/distanceToVisualDegreeRatio, attentionMaxDistance/distanceToVisualDegreeRatio)
                
                attentionSwitchFrequencyInSimulation = np.inf
                beliefUpdateFrequencyInSimulation = np.inf
                updateBeliefAndAttentionInSimulation = ba.UpdateBeliefAndAttentionState(attention, computePosterior, attentionSwitch, transferMultiAgentStatesToPositionDF,
                        attentionSwitchFrequencyInSimulation, beliefUpdateFrequencyInSimulation, burnTime)

                attentionSwitchFrequencyInPlay = int(0.2 * numMDPTimeStepPerSecond)
                beliefUpdateFrequencyInPlay = int(0.2 * numMDPTimeStepPerSecond)
                updateBeliefAndAttentionInPlay = ba.UpdateBeliefAndAttentionState(attention, computePosterior, attentionSwitch, transferMultiAgentStatesToPositionDF, 
                        attentionSwitchFrequencyInPlay, beliefUpdateFrequencyInPlay, burnTime)

                updatePhysicalStateByBeliefFrequencyInSimulationRoot = int(0.2 * numMDPTimeStepPerSecond)
                updatePhysicalStateByBeliefInSimulationRoot = ba.UpdatePhysicalStateImagedByBelief(updatePhysicalStateByBeliefFrequencyInSimulationRoot,
                        softParaForIdentity, softParaForSubtlety)
                reUpdatePhysicalStateByBeliefInSimulationRoot = ba.UpdatePhysicalStateImagedByBelief(updatePhysicalStateByBeliefFrequencyInSimulationRoot,
                        softParaForIdentity = 1, softParaForSubtlety = 1)
                updatePhysicalStateByBeliefFrequencyInSimulation = np.inf
                #updatePhysicalStateByBeliefInSimulation = ba.UpdatePhysicalStateImagedByBelief(updatePhysicalStateByBeliefFrequencyInSimulation,
                #        softParaForIdentity, softParaForSubtlety)
                updatePhysicalStateByBeliefInSimulation = lambda state: state
                
                updatePhysicalStateByBeliefFrequencyInPlay = np.inf
                #updatePhysicalStateByBeliefInPlay = ba.UpdatePhysicalStateImagedByBelief(updatePhysicalStateByBeliefFrequencyInPlay,
                #        softParaForIdentity, softParaForSubtlety)
                updatePhysicalStateByBeliefInPlay = lambda state: state

                transitionFunctionInSimulation = env.TransitionFunction(resetPhysicalState, resetBeliefAndAttention, updatePhysicalState, transiteStateWithoutActionChangeInSimulation, 
                        updateBeliefAndAttentionInSimulation, updatePhysicalStateByBeliefInSimulation)

                transitionFunctionInPlay = env.TransitionFunction(resetPhysicalState, resetBeliefAndAttention, updatePhysicalState, transiteStateWithoutActionChangeInPlay, 
                        updateBeliefAndAttentionInPlay, updatePhysicalStateByBeliefInPlay)
                
                numActionSpace = 4
                actionInterval = int(360/(numActionSpace))
                actionMagnitude = actionRatio * minSheepSpeed * numFramePerSecond
                actionSpaceFull = [(np.cos(degreeInPolar) * actionMagnitude, np.sin(degreeInPolar) * actionMagnitude) 
                        for degreeInPolar in np.arange(0, 360, actionInterval)/180 * math.pi] 
                actionSpaceHalf = [(np.cos(degreeInPolar) * actionMagnitude * 0.5, np.sin(degreeInPolar) * actionMagnitude * 0.5) 
                        for degreeInPolar in np.arange(0, 360, actionInterval)/180 * math.pi] 
                actionSpace = [(0, 0)] + actionSpaceFull + actionSpaceHalf
                getActionPrior = lambda state : {action: 1/len(actionSpace) for action in actionSpace}
                
                maxRollOutSteps = 5
                aliveBouns = 1/maxRollOutSteps
                deathPenalty = -1
                rewardFunction = reward.RewardFunctionTerminalPenalty(sheepId, aliveBouns, actionCost, deathPenalty, isTerminal, actionSpace)  
                rewardRollout = lambda state, action, nextState: rewardFunction(state, action) 


                cInit = 1
                #cBase = 50
                scoreChild = ScoreChild(cInit, cBase)
                selectAction = SelectAction(scoreChild)
                selectNextState = SelectNextState(selectAction)
                
                initializeChildren = InitializeChildren(actionSpace, transitionFunctionInSimulation, getActionPrior)
                expand = Expand(isTerminal, initializeChildren)
                pWidening = PWidening(alpha, C)
                expandNewState = ExpandNextState(transitionFunctionInSimulation, pWidening)
                
                rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
                rolloutHeuristic = lambda state: 0
                estimateValue = RollOut(rolloutPolicy, maxRollOutSteps, transitionFunctionInSimulation, rewardRollout, isTerminal, rolloutHeuristic)
               
                numActionPlaned = 1
                outputAction = OutputAction(numActionPlaned, actionSpace)
                #numSimulations = int(numTotalSimulationTimes/numTree)
                
                #sheepColorInMcts = np.array([0, 255, 0])
                #wolfColorInMcts = np.array([255, 0, 0])
                #distractorColorInMcts = np.array([255, 255, 255])
                #saveImageMCTS = True
                #mctsRender = env.MctsRender(numAgent, screen, xBoundary[1], yBoundary[1], screenColor, sheepColorInMcts, wolfColorInMcts, distractorColorInMcts, circleSize, saveImageMCTS, saveImageFile)
                #mctsRenderOn = False
                #mctsRender = None
                #pg.init()
                #mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectAction, mctsRender, mctsRenderOn)
                pwMultipleTrees = PWMultipleTrees(numSimulations, selectAction, selectNextState, expand, expandNewState, estimateValue, backup, outputAction)
                
                maxRunningSteps = int(25 * numMDPTimeStepPerSecond)
                makeDiffSimulationRoot = MakeDiffSimulationRoot(isTerminal, updatePhysicalStateByBeliefInSimulationRoot, reUpdatePhysicalStateByBeliefInSimulationRoot)
                runMCTSTrjactory = RunMCTSTrjactory(maxRunningSteps, numTree, numActionPlaned, sheepActionUpdateFrequency, transitionFunctionInPlay, isTerminal, makeDiffSimulationRoot, render)

                rootAction = actionSpace[np.random.choice(range(numActionSpace))]
                numTrial = 10
                trajectories = [runMCTSTrjactory(pwMultipleTrees) for trial in range(numTrial)]
               
                savePath = getSavePath({'chasingSubtlety': chasingSubtlety, 'subIndex': subIndex})
                tsl.saveToPickle(trajectories, savePath)
                getCSVSavePath = self.getCSVSavePathByCondition(condition)
                
                startStatsIndex = 1
                def getTrueWolfIdAcc(trajectory):
                    AccTrial = []
                    for timeStepIndex in range(len(trajectory) - 2):
                        timeStep = trajectory[timeStepIndex]
                        wolfId = timeStep[0][0][3][0]
                        wolfSubtlety = timeStep[0][0][3][1]
                        #print(wolfId, '**', wolfIdInEach)
                        if timeStepIndex >= startStatsIndex:
                            IdAcc = np.mean([int(IdAndSubtlety[0] == wolfId) for IdAndSubtlety in timeStep[5]])
                            AccTrial.append(IdAcc)
                    meanAcc = np.mean(AccTrial)
                    return meanAcc
                meanIdentiy = np.mean([getTrueWolfIdAcc(trajectory) for trajectory in trajectories])
                meanIdentiyOnConditions.update({chasingSubtlety: meanIdentiy})
                
                def getTrueWolfIdSubtletyAcc(trajectory):
                    AccTrial = []
                    for timeStepIndex in range(len(trajectory) - 2):
                        timeStep = trajectory[timeStepIndex]
                        wolfId = timeStep[0][0][3][0]
                        wolfSubtlety = timeStep[0][0][3][1]
                        #print(wolfId, '**', wolfIdInEach)
                        if timeStepIndex >= startStatsIndex:
                            IdAndSubtletyAcc = np.mean([int((IdAndSubtlety[0] == wolfId) and (IdAndSubtlety[1] == wolfSubtlety)) for IdAndSubtlety in timeStep[5]])
                            AccTrial.append(IdAndSubtletyAcc)
                    meanAcc = np.mean(AccTrial)
                    return meanAcc
                meanPerception = np.mean([getTrueWolfIdSubtletyAcc(trajectory) for trajectory in trajectories])
                meanPerceptionOnConditions.update({chasingSubtlety: meanPerception})
                
                def getActionDeviationLevel(trajectory):
                    AccTrial = []
                    for timeStepIndex in range(len(trajectory) - 2):
                        timeStep = trajectory[timeStepIndex]
                        actionReal = np.array(timeStep[1])
                        actionOnTruth = np.array(timeStep[4])
                        if timeStepIndex >= startStatsIndex:
                            deviateLevel = round(agf.computeAngleBetweenVectors(actionReal, actionOnTruth) / (math.pi / 4))
                            AccTrial.append(deviateLevel)
                    meanAcc = np.mean(AccTrial)
                    return meanAcc
                meanAction = np.mean([getActionDeviationLevel(trajectory) for trajectory in trajectories])
                meanActionOnConditions.update({chasingSubtlety: meanAction})
                
                def getVelocityDiff(trajectory):
                    AccTrial = []
                    for timeStepIndex in range(len(trajectory) - 2):
                        timeStep = trajectory[timeStepIndex]
                        velReal = np.array(timeStep[0][0][0][1][0])
                        velWithActionOnTruth = np.array(timeStep[2][1][0])
                        velWithActionOppo = np.array(timeStep[3][1][0])
                        if timeStepIndex >= startStatsIndex:
                            velDiffNormWithActionOnTruth = np.linalg.norm((velReal - velWithActionOnTruth))
                            velDiffNormWithActionOppo = np.linalg.norm((velReal - velWithActionOppo))
                            velDiffRatio = 1.0 * velDiffNormWithActionOnTruth / velDiffNormWithActionOppo 
                            AccTrial.append(velDiffRatio)
                    meanAcc = np.mean(AccTrial)
                    return meanAcc
                meanVelDiff = np.mean([getVelocityDiff(trajectory) for trajectory in trajectories])
                meanVelDiffOnConditions.update({chasingSubtlety: meanVelDiff})
            
                getEscapeAcc = lambda trajectory: int(len(trajectory) >= (maxRunningSteps - 2))
                meanEscape = np.mean([getEscapeAcc(trajectory) for trajectory in trajectories])
                meanEscapeOnConditions.update({chasingSubtlety: meanEscape})
            
            
            allResults.append(meanEscapeOnConditions)
            results = pd.DataFrame(allResults)
            escapeCSVSavePath = getCSVSavePath({'measure': 'escape'})
            results.to_csv(escapeCSVSavePath)
            
            allIdentityResults.append(meanIdentiyOnConditions)
            identityResults = pd.DataFrame(allIdentityResults)
            identityCSVSavePath = getCSVSavePath({'measure': 'identity'})
            identityResults.to_csv(identityCSVSavePath)
            
            allPerceptionResults.append(meanPerceptionOnConditions)
            perceptionResults = pd.DataFrame(allPerceptionResults)
            perceptionCSVSavePath = getCSVSavePath({'measure': 'percetion'})
            perceptionResults.to_csv(perceptionCSVSavePath)
             
            allActionResults.append(meanActionOnConditions)
            actionResults = pd.DataFrame(allActionResults)
            actionCSVSavePath = getCSVSavePath({'measure': 'action'})
            actionResults.to_csv(actionCSVSavePath)
            
            allVelDiffResults.append(meanVelDiffOnConditions)
            velDiffResults = pd.DataFrame(allVelDiffResults)
            velDiffCSVSavePath = getCSVSavePath({'measure': 'velDiff'})
            velDiffResults.to_csv(velDiffCSVSavePath)

def drawPerformanceline(dataDf, axForDraw):
    plotDf = dataDf.reset_index() 
    plotDf.plot(x = "subtleties", y = "escapeRate", ax = axForDraw)

def main():     
    manipulatedVariables = OrderedDict()
    manipulatedVariables['alpha'] = [0.25]
    #manipulatedVariables['attType'] = ['idealObserver']#, 'hybrid4']
    manipulatedVariables['attType'] = ['hybrid4']#, 'preAttention']
    #manipulatedVariables['attType'] = ['preAttention']
    #manipulatedVariables['attType'] = ['idealObserver', 'preAttention', 'attention4', 'hybrid4']
    #manipulatedVariables['attType'] = ['preAttentionMem0.65', 'preAttentionMem0.25', 'preAttentionPre0.5', 'preAttentionPre4.5']
    manipulatedVariables['C'] = [2]
    manipulatedVariables['minAttDist'] = [10.0, 40.0]#[10.0, 20.0, 40.0]
    manipulatedVariables['rangeAtt'] = [10.0]
    manipulatedVariables['cBase'] = [50]
    manipulatedVariables['numTrees'] = [4]
    manipulatedVariables['numSim'] = [185]
    manipulatedVariables['actRatio'] = [0.1, 0.9, 1.7]
    manipulatedVariables['burnTime'] = [0]
    manipulatedVariables['softId'] = [1]
    manipulatedVariables['softSubtlety'] = [1]
    manipulatedVariables['actCost'] = [0.0, 0.1, 0.5]
    manipulatedVariables['damp'] = [0.0, 0.5, 1.0]
 
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', 'data', 'mcts',
                                'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryExtension = '.pickle'
    getTrajectorySavePathByCondition = lambda condition: tsl.GetSavePath(trajectoryDirectory, trajectoryExtension, condition)
    measurementEscapeExtension = '.csv'
    getCSVSavePathByCondition = lambda condition: tsl.GetSavePath(trajectoryDirectory, measurementEscapeExtension, condition)
    runOneCondition = RunOneCondition(getTrajectorySavePathByCondition, getCSVSavePathByCondition)

    #runOneCondition(parametersAllCondtion[0])
    numCpuCores = os.cpu_count()
    numCpuToUse = int(numCpuCores)
    runPool = mp.Pool(numCpuToUse)
    runPool.map(runOneCondition, parametersAllCondtion)
   
    precisionToSubtletyDict={500:0,50:5,11:30,3.3:60,1.83:90,0.92:120,0.31:150,0.001: 180}
    
    #levelValues = [numTotalSimulationTimes, numTrees, subtleties]
    #levelNames = ["numTotalSimulationTimes", "numTrees", "subtleties"]

    #modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)

    #toSplitFrame = pd.DataFrame(index = modelIndex)

    #modelResultDf = toSplitFrame.groupby(levelNames).apply(evaluateMCTSInEscape)
    #fig = plt.figure()
    #plotLevels =levelNames [:1]
    #plotRowNum = len(levelValues[0])
    #plotColNum = len(levelValues[1])
    #plotCounter = 1

    #for (key, dataDf) in modelResultDf.groupby(["numTotalSimulationTimes", "numTrees"]):
    #    axForDraw = fig.add_subplot(plotRowNum, plotColNum, plotCounter)
    #    drawPerformanceline(dataDf, axForDraw)
    #    plotCounter+=1

    #plt.show()
    #fig.savefig('escapeRateActUpdateSheep2Wolf3Att95Tree3Sime36.png')
    #print("Finished evaluating")

if __name__ == "__main__":
    main()
    
