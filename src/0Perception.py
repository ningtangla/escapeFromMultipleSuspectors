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
from algorithms.stochasticPW import ScoreChild, SelectAction, SelectNextState, InitializeChildren, Expand, ExpandNextState, \
        PWidening, RollOut, backup, OutputAction, PWMultipleTrees

from simple1DEnv import TransitionFunction, RewardFunction, Terminal
from visualize import draw
import stochasticAgentsMotionSimulationByAccerelationActionBurnTime as ag
import Attention
import calPosteriorModifyForIdeal
import stochasticBeliefAndAttentionSimulationBurnTime as ba
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
            physicalState, beliefAndAttention = currState
            hypothesisInformation, positionOldTimeDF = beliefAndAttention
            probabilityOnHypothesisAttention = np.exp(hypothesisInformation['logP']) 
            posteriorOnHypothesisAttention = probabilityOnHypothesisAttention/probabilityOnHypothesisAttention.sum()
            probabilityOnAttentionSlotByGroupbySum = posteriorOnHypothesisAttention.groupby(['wolfIdentity','sheepIdentity']).sum().values
            posterior = probabilityOnAttentionSlotByGroupbySum/np.sum(probabilityOnAttentionSlotByGroupbySum)
            stateToRecord = [physicalState, posterior]
            trajectory.append([stateToRecord, action]) 
            nextState = self.transitionFunctionInPlay(currState, action)
            #print('***', physicalState[0][0], action)
            #print('***', np.linalg.norm(physicalState[1][0]), np.linalg.norm(physicalState[1][10]))
            currState = nextState
        return trajectory

class RunOneCondition:
    def __init__(self, getTrajectorySavePathByCondition, getCSVSavePathByCondition):
        self.getTrajectorySavePathByCondition = getTrajectorySavePathByCondition
        self.getCSVSavePathByCondition = getCSVSavePathByCondition
    
    def __call__(self, condition):
        
        getSavePath = self.getTrajectorySavePathByCondition(condition)
        attentionType = condition['attentionType']
        alpha = condition['alphaForStateWidening']
        C = condition['CForStateWidening']
        minAttentionDistance = condition['minAttentionDistance']
        rangeAttention = condition['rangeAttention']
        numTree = condition['numTrees']
        numSimulations = condition['numSimulationTimes']
        actionRatio = condition['actionRatio']
        cBase = condition['cBase']
        burnTime = condition['burnTime']

        numSub = 2
        allResults = []
        possibleTrialSubtleties = [1.83, 0.92, 0.31, 0.001]
        for subIndex in range(numSub):
            meanIdentityPerceptionOnConditions = {}
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
                sheepPolicy = ag.SheepPolicy(sheepActionUpdateFrequency, minSheepSpeed, maxSheepSpeed, warmUpTimeSteps, burnTime)
                
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
               
                minDistance = 0.0 * distanceToVisualDegreeRatio
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
                possibleSubtleties = [500.0, 11.0, 3.3, 1.83, 0.92, 0.31, 0.001]
                resetBeliefAndAttention = ba.ResetBeliefAndAttention(sheepId, suspectorIds, possibleSubtleties, attentionLimitation, transferMultiAgentStatesToPositionDF, attention)
               
                maxAttentionDistance = minAttentionDistance + rangeAttention
                attentionMinDistance = minAttentionDistance * distanceToVisualDegreeRatio
                attentionMaxDistance = maxAttentionDistance * distanceToVisualDegreeRatio
                numStandardErrorInDistanceRange = 4
                calDistancePriorOnAttentionSlot = Attention.CalDistancePriorOnAttentionSlot(attentionMinDistance, attentionMaxDistance, numStandardErrorInDistanceRange)
                attentionSwitch = Attention.AttentionSwitch(attentionLimitation, calDistancePriorOnAttentionSlot)    
                computePosterior = calPosteriorModifyForIdeal.CalPosteriorLog(minDistance)

                print(attentionLimitation, attentionMinDistance/distanceToVisualDegreeRatio, attentionMaxDistance/distanceToVisualDegreeRatio)
                
                attentionSwitchFrequencyInSimulation = np.inf
                beliefUpdateFrequencyInSimulation = np.inf
                updateBeliefAndAttentionInSimulation = ba.UpdateBeliefAndAttentionState(attention, computePosterior, attentionSwitch, transferMultiAgentStatesToPositionDF,
                        attentionSwitchFrequencyInSimulation, beliefUpdateFrequencyInSimulation, burnTime)

                attentionSwitchFrequencyInPlay = int(0.6 * numMDPTimeStepPerSecond)
                beliefUpdateFrequencyInPlay = int(0.2 * numMDPTimeStepPerSecond)
                updateBeliefAndAttentionInPlay = ba.UpdateBeliefAndAttentionState(attention, computePosterior, attentionSwitch, transferMultiAgentStatesToPositionDF, 
                        attentionSwitchFrequencyInPlay, beliefUpdateFrequencyInPlay, burnTime)

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
                rewardRollout = lambda state, action, nextState: rewardFunction(state, action) 

                numActionSpace = 8
                actionInterval = int(360/(numActionSpace))
                actionMagnitude = actionRatio * minSheepSpeed
                actionSpace = [(np.cos(degreeInPolar) * actionMagnitude, np.sin(degreeInPolar) * actionMagnitude) for degreeInPolar in np.arange(0, 360, actionInterval)/180 * math.pi] 
                getActionPrior = lambda state : {action: 1/len(actionSpace) for action in actionSpace}

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
                makeDiffSimulationRoot = MakeDiffSimulationRoot(isTerminal, updatePhysicalStateByBeliefInSimulationRoot)
                runMCTSTrjactory = RunMCTSTrjactory(maxRunningSteps, numTree, numActionPlaned, sheepActionUpdateFrequency, transitionFunctionInPlay, isTerminal, makeDiffSimulationRoot, render)

                rootAction = actionSpace[np.random.choice(range(numActionSpace))]
                numTrial = 10
                trajectories = [runMCTSTrjactory(pwMultipleTrees) for trial in range(numTrial)]
               
                savePath = getSavePath({'chasingSubtlety': chasingSubtlety, 'subIndex': subIndex})
               # tsl.saveToPickle(trajectories, savePath)
               # def getTrueWolfIndentityAcc(trajectory):
               #     AccTrial = []
               #     for timeStepIndex in range(len(trajectory)):
               #         timeStep = trajectory[timeStepIndex]
               #         wolfId = trajectory[0][0][0][3][0]
               #         wolfIdInEach = timeStep[0][0][3][0]
               #         #print(wolfId, '**', wolfIdInEach)
               #         if (timeStepIndex % 3 == 0) and timeStepIndex >= 11:
               #             AccTrial.append(timeStep[0][1][int(wolfIdInEach) - 1])
               #     #meanIdentityAcc = np.mean(AccTrial)
               #     meanIdentityAcc = np.mean(np.array([timeStep[0][1][int(timeStep[0][0][3][0] - 1)] for timeStep in trajectory])[11:])
               #     return meanIdentityAcc
                getTrueWolfIndentityAcc = lambda trajectory: np.mean(np.array([timeStep[0][1][int(timeStep[0][0][3][0] - 1)] for timeStep in trajectory])[11:])
                meanIdentityPerception = np.mean([getTrueWolfIndentityAcc(trajectory) for trajectory in trajectories])
                meanIdentityPerceptionOnConditions.update({chasingSubtlety: meanIdentityPerception})
                print(meanIdentityPerceptionOnConditions)
            allResults.append(meanIdentityPerceptionOnConditions)
            results = pd.DataFrame(allResults)
            getCSVSavePath = self.getCSVSavePathByCondition(condition)
            csvSavePath = getCSVSavePath({})
            results.to_csv(csvSavePath)

def drawPerformanceline(dataDf, axForDraw):
    plotDf = dataDf.reset_index() 
    plotDf.plot(x = "subtleties", y = "escapeRate", ax = axForDraw)

def main():     
    manipulatedVariables = OrderedDict()
    manipulatedVariables['alphaForStateWidening'] = [0.25]
    #manipulatedVariables['attentionType'] = ['idealObserver', 'hybrid4']
    #manipulatedVariables['attentionType'] = ['hybrid4']
    #manipulatedVariables['attentionType'] = ['preAttention']
    manipulatedVariables['attentionType'] = ['idealObserver', 'preAttention', 'attention4', 'hybrid4']
    #manipulatedVariables['attentionType'] = ['preAttentionMem0.65', 'preAttentionMem0.25', 'preAttentionPre0.5', 'preAttentionPre4.5']
    manipulatedVariables['CForStateWidening'] = [2]
    manipulatedVariables['minAttentionDistance'] = [40.0]
    manipulatedVariables['rangeAttention'] = [4]
    manipulatedVariables['cBase'] = [50]
    manipulatedVariables['numTrees'] = [2]
    manipulatedVariables['numSimulationTimes'] = [1]
    manipulatedVariables['actionRatio'] = [0.2]
    manipulatedVariables['burnTime'] = [1]
 
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
    
