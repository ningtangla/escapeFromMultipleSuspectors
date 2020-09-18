import pygame as pg
import numpy as np
import os
import functools as ft

class DrawBackground:
    def __init__(self, screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth, xObstacles = None, yObstacles = None):
        self.screen = screen
        self.screenColor = screenColor
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.lineColor = lineColor
        self.lineWidth = lineWidth
        self.xObstacles = xObstacles
        self.yObstacles = yObstacles

    def __call__(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    exit()
        self.screen.fill(self.screenColor)
        rectPos = [self.xBoundary[0], self.yBoundary[0], self.xBoundary[1], self.yBoundary[1]]
        pg.draw.rect(self.screen, self.lineColor, rectPos, self.lineWidth)
        if self.xObstacles and self.yObstacles:
            for xObstacle, yObstacle in zip(self.xObstacles, self.yObstacles):
                rectPos = [xObstacle[0], yObstacle[0], xObstacle[1] - xObstacle[0], yObstacle[1] - yObstacle[0]]
                pg.draw.rect(self.screen, self.lineColor, rectPos)
        return

class DrawCircleOutside:
    def __init__(self, screen, outsideCircleAgentIds, positionIndex, circleColors, circleSize):
        self.screen = screen
        self.outsideCircleAgentIds = outsideCircleAgentIds
        self.xIndex, self.yIndex = positionIndex
        self.circleColors = circleColors
        self.circleSize = circleSize

    def __call__(self, state):
        for agentIndex in self.outsideCircleAgentIds:
            agentPos = [np.int(state[agentIndex][self.xIndex]), np.int(state[agentIndex][self.yIndex])]
            agentColor = tuple(self.circleColors[list(self.outsideCircleAgentIds).index(agentIndex)])
            pg.draw.circle(self.screen, agentColor, agentPos, self.circleSize)
        return

class DrawState:
    def __init__(self, fps, screen, colorSpace, circleSize, agentIdsToDraw, positionIndex, saveImage, imagePath, 
            drawBackGround, updateColorByPosterior = None, drawCircleOutside = None):
        self.fps = fps
        self.screen = screen
        self.colorSpace = colorSpace
        self.circleSize = circleSize
        self.agentIdsToDraw = agentIdsToDraw
        self.xIndex, self.yIndex = positionIndex
        self.saveImage = saveImage
        self.imagePath = imagePath
        self.drawBackGround = drawBackGround
        self.updateColorByPosterior = updateColorByPosterior
        self.drawCircleOutside = drawCircleOutside

    def __call__(self, state, posterior = None):
        fpsClock = pg.time.Clock()
        
        self.drawBackGround()
        if (posterior is not None) and self.updateColorByPosterior:
            circleColors = self.updateColorByPosterior(self.colorSpace, posterior)
        else:
            circleColors = self.colorSpace
        if self.drawCircleOutside:
            self.drawCircleOutside(state)
        for agentIndex in self.agentIdsToDraw:
            agentPos = [np.int(state[agentIndex][self.xIndex]), np.int(state[agentIndex][self.yIndex])]
            agentColor = tuple(circleColors[agentIndex])
            pg.draw.circle(self.screen, agentColor, agentPos, self.circleSize)

        pg.display.flip()
        
        if self.saveImage == True:
            filenameList = os.listdir(self.imagePath)
            pg.image.save(self.screen, self.imagePath + '/' + str(len(filenameList))+'.png')
        
        fpsClock.tick(self.fps)
        return self.screen

class InterpolateState:
    def __init__(self, numFramesToInterpolate):
        self.numFramesToInterpolate = numFramesToInterpolate
    def __call__(self, state, nextState):
        interpolatedStates = [state]
        actionForInterpolation = (np.array(nextState) - np.array(state)) / (self.numFramesToInterpolate + 1) 
        for frameIndex in range(self.numFramesToInterpolate):
            nextStateForInterpolation = np.array(state) + np.array(actionForInterpolation)
            interpolatedStates.append(nextStateForInterpolation)
            state = nextStateForInterpolation
            #actionForInterpolation = nextActionForInterpolation
        print('***', np.array(interpolatedStates)[:, 0], np.array(actionForInterpolation)[0])
        return interpolatedStates

class ChaseTrialWithTraj:
    def __init__(self, stateIndex, drawState, interpolateState = None, actionIndex = None, posteriorIndex = None):
        self.stateIndex = stateIndex
        self.drawState = drawState
        self.interpolateState = interpolateState
        self.actionIndex = actionIndex
        self.posteriorIndex = posteriorIndex

    def __call__(self, trajectory):
        for timeStepIndex in range(len(trajectory) - 1):
            stateInSheepMind = trajectory[timeStepIndex][0]
            physicalState, posteriorInState = stateInSheepMind
            agentStates, agentActions, timeStep, wolfIdAndSubtlety = physicalState
            #wolfId, wolfSubtlety = wolfIdAndSubtlety
            nextStateInSheepMind = trajectory[timeStepIndex+1][0]
            nextPhysicalState, nextPosteriorInState = nextStateInSheepMind
            nextAgentStates, nextAgentActions, nextTimeStep, nextWolfIdAndSubtlety = nextPhysicalState
            state = agentStates
            nextState = nextAgentStates
            if self.posteriorIndex:
                posterior = posteriorInState
            else:
                posterior = None
            if self.interpolateState and timeStepIndex!= len(trajectory) - 1:
                statesToDraw = self.interpolateState(state, nextState)
            else:
                statesToDraw  = [state]
            for state in statesToDraw:
                screen = self.drawState(state, posterior)
        return
