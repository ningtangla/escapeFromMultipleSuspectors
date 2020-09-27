import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import pathos.multiprocessing as mp
import pygame as pg
from pygame.color import THECOLORS

from drawDemo import DrawBackground, DrawCircleOutside, DrawState, ChaseTrialWithTraj, InterpolateState, DrawPlanningAna
from trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
        GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from stochasticAgentsMotionSimulationByAccerelationActionBurnTime import CheckBoundaryAndAdjust, TransiteMultiAgentMotion

def updateColorSpace(colorSpace, posterior):
    wolfColors = [np.maximum([0, 0, 0], np.minimum([255, 255, 255], np.ones(3) * 255 * (1 - 1.5*wolfBelief) + np.array([255, 0, 0]) * 1.5*wolfBelief)) for wolfBelief in posterior]
    updatedColorSpace = np.array([np.array([0, 255, 0])] + wolfColors)
    return updatedColorSpace

def main():
    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', 'data', 'mcts',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    
    alphaForStateWidening = 0.25
    CForStateWidening = 2
    cBase = 50
    numTrees = 1
    numSimulationTimes = 151
    trajectoryFixedParameters = {'alphaForStateWidening': alphaForStateWidening, 'CForStateWidening': CForStateWidening,
            'cBase': cBase, 'numTrees': numTrees, 'numSimulationTimes': numSimulationTimes}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    minAttentionDistance = 40
    rangeAttention = 10.0
    actionRatio = 0.2
    burnTime = 0
    softId = 1
    softSubtlety = 1
    trajectoryParameters = {'minAttentionDistance': minAttentionDistance, 'rangeAttention': rangeAttention, 'actionRatio': actionRatio, 
            'burnTime': burnTime, 'softId': softId, 'softSubtlety': softSubtlety}
    chasingSubtlety = 500.0
    subIndex = 0
    attentionType = 'idealObserver'
    #attentionType = 'preAttention'
    #attentionType = 'hybrid4'
    trajectoryParameters.update({'chasingSubtlety': chasingSubtlety, 'subIndex': subIndex, 'attentionType': attentionType})

    trajectories = loadTrajectories(trajectoryParameters) 
    # generate demo image
    screenWidth = 640
    screenHeight = 480
    screen = pg.display.set_mode((screenWidth, screenHeight))
    screenColor = THECOLORS['black']
    xBoundary = [0, 640]
    yBoundary = [0, 480]
    lineColor = THECOLORS['white']
    lineWidth = 4
    drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)
    
    FPS = 2
    numSheep = 1
    numWolves = 24
    circleColorSpace = [[0, 255, 0]] * numSheep + [[255, 255, 255]] * numWolves
    circleSize = 10
    positionIndex = [0, 1]
    agentIdsToDraw = list(range(numSheep + numWolves))
    saveImage = True
    imageSavePath = os.path.join(trajectoryDirectory, 'picMovingSheep')
    if not os.path.exists(imageSavePath):
        os.makedirs(imageSavePath)
    imageFolderName = str('forDemo')
    saveImageDir = os.path.join(os.path.join(imageSavePath, imageFolderName))
    if not os.path.exists(saveImageDir):
        os.makedirs(saveImageDir)
    updateColorSpaceByPosterior = updateColorSpace
    
    #updateColorSpaceByPosterior = lambda originalColorSpace, posterior : originalColorSpace
    outsideCircleAgentIds = list(range(1, numSheep + numWolves))
    outsideCircleColor = np.array([[255, 255, 255]] * numWolves)
    outsideCircleSize = 1 
    drawCircleOutside = DrawCircleOutside(screen, outsideCircleAgentIds, positionIndex, outsideCircleColor, outsideCircleSize)
    drawState = DrawState(FPS, screen, circleColorSpace, circleSize, agentIdsToDraw, positionIndex, 
            saveImage, saveImageDir, drawBackground, updateColorSpaceByPosterior, drawCircleOutside)
    
    drawPlanningAna = DrawPlanningAna(FPS, screen, circleColorSpace, circleSize, agentIdsToDraw, positionIndex, 
            saveImage, saveImageDir, drawBackground, updateColorSpaceByPosterior, drawCircleOutside)
   # MDP Env
    xBoundary = [0,640]
    yBoundary = [0,480]
    #checkBoundaryAndAdjust = CheckBoundaryAndAdjust(xBoundary, yBoundary)
    #transiteInDemo = TransiteMultiAgentMotion(checkBoundaryAndAdjust)
    numFramesToInterpolate = int(FPS/5 - 1)
    interpolateState = InterpolateState(numFramesToInterpolate)
    
    stateIndexInTimeStep = 0
    actionIndexInTimeStep = 1
    posteriorIndexInTimeStep = 4
    chaseTrial = ChaseTrialWithTraj(stateIndexInTimeStep, drawState, interpolateState, 
            actionIndexInTimeStep, posteriorIndexInTimeStep, drawPlanningAna)
    
    print(len(trajectories))
    lens = [len(trajectory) for trajectory in trajectories]
    index = np.argsort(-np.array(lens))
    print(index)
    print(trajectories[0][1])
    [chaseTrial(trajectory) for trajectory in np.array(trajectories)[index[0:10]]]
    print([len(trajectory) for trajectory in np.array(trajectories)[index[:]]])
    #[chaseTrial(trajectory) for trajectory in np.array(trajectories)[index[13:14]]]

if __name__ == '__main__':
    main()
