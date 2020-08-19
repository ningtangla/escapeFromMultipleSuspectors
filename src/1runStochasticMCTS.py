import numpy as np
import pygame as pg
import pandas as pd
import pylab as plt
import itertools as it
import math
import os
import pathos.multiprocessing as mp

from anytree import AnyNode as Node
from anytree import RenderTree
from collections import OrderedDict

# Local import
from algorithms.stochasticMCTS import MCTS, CalculateScore, GetActionPrior, SelectAction, SelectChild, Expand, RollOut, backup, InitializeChildren

from simple1DEnv import TransitionFunction, RewardFunction, Terminal
from visualize import draw
import stochasticAgentsMotionSimulation as ag
import Attention
import calPosterior
import stochasticBeliefAndAttentionSimulation as ba
import env
import reward
import trajectoriesSaveLoad as tsl

class MakeDiffSimulationRoot():
    def __init__(self, isTerminal, updatePhysicalStateByBelief):
        self.isTerminal = isTerminal
        self.updatePhysicalStateByBelief = updatePhysicalStateByBelief
    def __call__(self, action, state):
        stateForSimulationRoot = self.updatePhysicalStateByBelief(state)
        while self.isTerminal(stateForSimulationRoot):
            stateForSimulationRoot = self.updatePhysicalStateByBelief(state)
        rootNode = Node(id={action: stateForSimulationRoot}, num_visited=0, sum_value=0, is_expanded = False)
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
        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            
            if self.isTerminal(currState):
                trajectory.append([currState, None])
                break
            if runningStep % (self.planFrequency * self.actionUpdateFrequecy) == 0:
                rootNodes = [self.makeDiffSimulationRoot(action, currState) for treeIndex in range(self.numTree)]
                actions = mcts(rootNodes)
                #actionSpace = [(np.cos(degreeInPolar), np.sin(degreeInPolar)) for degreeInPolar in np.arange(0, 360, 8)/180 * math.pi] 
                #actions = [actionSpace[np.random.choice(range(len(actionSpace)))]]
            action = actions[int(runningStep/self.actionUpdateFrequecy) % self.planFrequency]
            trajectory.append([currState, action]) 
            nextState = self.transitionFunctionInPlay(currState, action) 
            currState = nextState
        return trajectory

class RunOneCondition:
    def __init__(self, getTrajectorySavePathByCondition, getCSVSavePathByCondition):
        self.getTrajectorySavePathByCondition = getTrajectorySavePathByCondition
        self.getCSVSavePathByCondition = getCSVSavePathByCondition
    
    def __call__(self, condition):
        
        getSavePath = self.getTrajectorySavePathByCondition(condition)
        attentionType = condition['attentionType']
        minAttentionDistance = condition['minAttentionDistance']
        maxAttentionDistance = condition['maxAttentionDistance']
        numTree = condition['numTrees']
        numTotalSimulationTimes = condition['totalNumSimulationTimes']
        
        numSub = 10
        possibleSubtleties = [500, 11, 3.3, 1.83, 0.92, 0.31, 0.01]
        allResults = []
        for subIndex in range(numSub):
            meanEscapeOnConditions = {}
            for chasingSubtlety in possibleSubtleties: 

                print(numTree, chasingSubtlety, numTotalSimulationTimes, attentionType)
                numActionSpace = 8
                actionInterval = int(360/(numActionSpace))
                actionSpace = [(np.cos(degreeInPolar), np.sin(degreeInPolar)) for degreeInPolar in np.arange(0, 360, actionInterval)/180 * math.pi] 
                getActionPrior = GetActionPrior(actionSpace)

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
                sheepPolicy = ag.SheepPolicy(sheepActionUpdateFrequency, minSheepSpeed, maxSheepSpeed, warmUpTimeSteps)
                
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
                    precisionPerSlot=800.0
                    precisionForUntracked=800.0
                    memoryratePerSlot=1
                    memoryrateForUntracked=1
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
                attention = Attention.AttentionToPrecisionAndDecay(precisionPerSlot, precisionForUntracked, memoryratePerSlot, memoryrateForUntracked)    
                transferMultiAgentStatesToPositionDF = ba.TransferMultiAgentStatesToPositionDF(numAgent) 
                resetBeliefAndAttention = ba.ResetBeliefAndAttention(sheepId, suspectorIds, possibleSubtleties, attentionLimitation, transferMultiAgentStatesToPositionDF, attention)
               
                attentionMinDistance = minAttentionDistance * distanceToVisualDegreeRatio
                attentionMaxDistance = maxAttentionDistance * distanceToVisualDegreeRatio
                numStandardErrorInDistanceRange = 4
                calDistancePriorOnAttentionSlot = Attention.CalDistancePriorOnAttentionSlot(attentionMinDistance, attentionMaxDistance, numStandardErrorInDistanceRange)
                attentionSwitch = Attention.AttentionSwitch(attentionLimitation, calDistancePriorOnAttentionSlot)    
                computePosterior = calPosterior.CalPosteriorLog(minDistance)

                attentionSwitchFrequencyInSimulation = np.inf
                beliefUpdateFrequencyInSimulation = np.inf
                updateBeliefAndAttentionInSimulation = ba.UpdateBeliefAndAttentionState(attention, computePosterior, attentionSwitch, transferMultiAgentStatesToPositionDF,
                        attentionSwitchFrequencyInSimulation, beliefUpdateFrequencyInSimulation)

                attentionSwitchFrequencyInPlay = int(0.6 * numMDPTimeStepPerSecond)
                beliefUpdateFrequencyInPlay = int(0.2 * numMDPTimeStepPerSecond)
                updateBeliefAndAttentionInPlay = ba.UpdateBeliefAndAttentionState(attention, computePosterior, attentionSwitch, transferMultiAgentStatesToPositionDF, 
                        attentionSwitchFrequencyInPlay, beliefUpdateFrequencyInPlay)

                updatePhysicalStateByBeliefFrequencyInSimulationRoot = int(0.6 * numMDPTimeStepPerSecond)
                updatePhysicalStateByBeliefInSimulationRoot = ba.UpdatePhysicalStateImagedByBelief(updatePhysicalStateByBeliefFrequencyInSimulationRoot)
                updatePhysicalStateByBeliefFrequencyInSimulation = np.inf
                updatePhysicalStateByBeliefInSimulation = ba.UpdatePhysicalStateImagedByBelief(updatePhysicalStateByBeliefFrequencyInSimulation)
                
                updatePhysicalStateByBeliefFrequencyInPlay = np.inf
                updatePhysicalStateByBeliefInPlay = ba.UpdatePhysicalStateImagedByBelief(updatePhysicalStateByBeliefFrequencyInPlay)

                transitionFunctionInSimulation = env.TransitionFunction(resetPhysicalState, resetBeliefAndAttention, updatePhysicalState, transiteStateWithoutActionChangeInSimulation, 
                        updateBeliefAndAttentionInSimulation, updatePhysicalStateByBeliefInSimulation)

                transitionFunctionInPlay = env.TransitionFunction(resetPhysicalState, resetBeliefAndAttention, updatePhysicalState, transiteStateWithoutActionChangeInPlay, 
                        updateBeliefAndAttentionInPlay, updatePhysicalStateByBeliefInPlay)
                
                maxRollOutSteps = 5
                aliveBouns = 1/maxRollOutSteps
                deathPenalty = -1
                rewardFunction = reward.RewardFunctionTerminalPenalty(sheepId, aliveBouns, deathPenalty, isTerminal)  

                cInit = 1
                cBase = 30
                calculateScore = CalculateScore(cInit, cBase)
                selectChild = SelectChild(calculateScore)
                
                initializeChildren = InitializeChildren(actionSpace, transitionFunctionInSimulation, getActionPrior)
                expand = Expand(isTerminal, initializeChildren)

                rolloutPolicy = lambda state: actionSpace[np.random.choice(range(numActionSpace))]
                rollout = RollOut(rolloutPolicy, maxRollOutSteps, transitionFunctionInSimulation, rewardFunction, isTerminal)
               
                numActionPlaned = 1
                selectAction = SelectAction(numActionPlaned, actionSpace)
                numSimulations = int(numTotalSimulationTimes/numTree)
                
                #sheepColorInMcts = np.array([0, 255, 0])
                #wolfColorInMcts = np.array([255, 0, 0])
                #distractorColorInMcts = np.array([255, 255, 255])
                #saveImageMCTS = True
                #mctsRender = env.MctsRender(numAgent, screen, xBoundary[1], yBoundary[1], screenColor, sheepColorInMcts, wolfColorInMcts, distractorColorInMcts, circleSize, saveImageMCTS, saveImageFile)
                mctsRenderOn = False
                mctsRender = None
                pg.init()
                mcts = MCTS(numSimulations, selectChild, expand, rollout, backup, selectAction, mctsRender, mctsRenderOn)
                
                maxRunningSteps = int(25 * numMDPTimeStepPerSecond)
                makeDiffSimulationRoot = MakeDiffSimulationRoot(isTerminal, updatePhysicalStateByBeliefInSimulationRoot)
                runMCTSTrjactory = RunMCTSTrjactory(maxRunningSteps, numTree, numActionPlaned, sheepActionUpdateFrequency, transitionFunctionInPlay, isTerminal, makeDiffSimulationRoot, render)

                rootAction = actionSpace[np.random.choice(range(numActionSpace))]
                numTrial = 15
                print(attentionLimitation, attentionMinDistance/distanceToVisualDegreeRatio, attentionMaxDistance/distanceToVisualDegreeRatio)
                trajectories = [runMCTSTrjactory(mcts) for trial in range(numTrial)]
               
                savePath = getSavePath({'chasingSubtlety': chasingSubtlety, 'subIndex': subIndex})
                tsl.saveToPickle(trajectories, savePath)

                meanEscape = np.mean([1 if len(trajectory) >= (maxRunningSteps - 1) else 0 for trajectory in trajectories])
                meanEscapeOnConditions.update({chasingSubtlety: meanEscape})
            allResults.append(meanEscapeOnConditions)
            print(meanEscapeOnConditions)
            results = pd.DataFrame(allResults)
            getCSVSavePath = self.getCSVSavePathByCondition(condition)
            csvSavePath = getCSVSavePath({})
            results.to_csv(csvSavePath)

def drawPerformanceline(dataDf, axForDraw):
    plotDf = dataDf.reset_index() 
    plotDf.plot(x = "subtleties", y = "escapeRate", ax = axForDraw)

def main():     
    manipulatedVariables = OrderedDict()
    manipulatedVariables['attentionType'] = ['idealObserver', 'preAttention', 'attention3', 'hybrid3', 'attention4', 'hybrid4']
    manipulatedVariables['minAttentionDistance'] = [7.5]
    manipulatedVariables['maxAttentionDistance'] = [11.5, 15.5]
    manipulatedVariables['numTrees'] = [2]
    manipulatedVariables['totalNumSimulationTimes'] = [36, 72]#, 120]

 
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

    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.9 * numCpuCores)
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
    