import pygame as pg
import numpy as np
import os
import functools as ft
import math

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
        #print('***', np.array(interpolatedStates)[:, 0], np.array(actionForInterpolation)[0])
        return interpolatedStates

class DrawPlanningAna:
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
        self.precisionToSubtletyDict = {500.0:5,50.0:5,11.0:30,3.3:60,1.83:90,0.92:120,0.31:150,0.001:179}

    def __call__(self, timeStep, nextTimeStep):
        fpsClock = pg.time.Clock()
        state = timeStep[0][0][0]
        nextVelocity = nextTimeStep[0][0][1]
        
        truthId = timeStep[0][0][3][0]
        truthSubtlety = self.precisionToSubtletyDict[timeStep[0][0][3][1]]

        sampledId = timeStep[5][0][0]
        sampledSubtlety = self.precisionToSubtletyDict[timeStep[5][0][1]]

        action = timeStep[1]
        actionOnTruth = timeStep[4]

        nextSheepVel = nextVelocity[0]
        nextSheepVelOnTruth = nextTimeStep[2][1][0]
        
        self.drawBackGround()
        for agentIndex in self.agentIdsToDraw:
            agentPos = [np.int(state[agentIndex][self.xIndex]), np.int(state[agentIndex][self.yIndex])]
            agentColor = tuple(self.colorSpace[agentIndex])
            pg.draw.circle(self.screen, agentColor, agentPos, self.circleSize)

        sheepPos = state[0]
        
        sampledPos = state[sampledId]
        heatSeekingSampled = (sheepPos - sampledPos) / np.linalg.norm((sheepPos - sampledPos))# * (self.circleSize + 15)
        pg.draw.circle(self.screen, [255, 255, 0], [int(dim) for dim in sampledPos], self.circleSize + 5, 5)
        centerAngeledSampled = np.arctan2(heatSeekingSampled[1], heatSeekingSampled[0])
        starAngleSampled = - centerAngeledSampled - sampledSubtlety / 180 * math.pi
        endAngleSampled = - centerAngeledSampled + sampledSubtlety / 180 * math.pi
        pg.draw.arc(self.screen, [255, 255, 0], [int(sampledPos[0]) - 28, int(sampledPos[1]) - 28, 56, 56], starAngleSampled, endAngleSampled, 7) 
        
        truthPos = state[truthId]
        heatSeekingTruth = (sheepPos - truthPos) / np.linalg.norm((sheepPos - truthPos))# * (self.circleSize + 10)
        pg.draw.circle(self.screen, [255, 0, 0], [int(dim) for dim in truthPos], self.circleSize + 3, 3)
        centerAngeledTruth = np.arctan2(heatSeekingTruth[1], heatSeekingTruth[0])
        starAngleTruth = - centerAngeledTruth - truthSubtlety / 180 * math.pi 
        endAngleTruth = - centerAngeledTruth + truthSubtlety / 180 * math.pi
        pg.draw.arc(self.screen, [255, 0, 0], [int(truthPos[0]) - 23, int(truthPos[1]) - 23, 46, 46], starAngleTruth, endAngleTruth, 7) 

        actionLine = np.array(action) / (np.linalg.norm(action) + 1e-12) * (self.circleSize + 20)
        pg.draw.line(self.screen, [255, 255, 0], [int(dim) for dim in sheepPos], [int(dim) for dim in sheepPos + actionLine], 5)
        
        actionTruthLine = np.array(actionOnTruth) / (np.linalg.norm(actionOnTruth) + 1e-12) * (self.circleSize + 15)
        pg.draw.line(self.screen, [255, 0, 0], [int(dim) for dim in sheepPos], [int(dim) for dim in sheepPos + actionTruthLine], 5)

        pg.display.flip()
        
        for _ in range(10):
            if self.saveImage == True:
                filenameList = os.listdir(self.imagePath)
                pg.image.save(self.screen, self.imagePath + '/' + str(len(filenameList))+'.png')
        velLine = np.array(nextSheepVel) / (np.linalg.norm(nextSheepVel) + 1e-12) * (self.circleSize + 35)
        pg.draw.line(self.screen, [255, 255, 0], [int(dim) for dim in sheepPos], [int(dim) for dim in sheepPos + velLine], 10)
        
        velTruthLine = np.array(nextSheepVelOnTruth) / (np.linalg.norm(nextSheepVelOnTruth) + 1e-12) * (self.circleSize + 25)
        pg.draw.line(self.screen, [255, 0, 0], [int(dim) for dim in sheepPos], [int(dim) for dim in sheepPos + velTruthLine], 10)
        
        pg.display.flip()
        
        for _ in range(10):
            if self.saveImage == True:
                filenameList = os.listdir(self.imagePath)
                pg.image.save(self.screen, self.imagePath + '/' + str(len(filenameList))+'.png')

        fpsClock.tick(self.fps)
        return self.screen

class ChaseTrialWithTraj:
    def __init__(self, stateIndex, drawState, interpolateState = None, actionIndex = None, posteriorIndex = None, drawPlanningAna = None):
        self.stateIndex = stateIndex
        self.drawState = drawState
        self.interpolateState = interpolateState
        self.actionIndex = actionIndex
        self.posteriorIndex = posteriorIndex
        self.drawPlanningAna = drawPlanningAna

    def __call__(self, trajectory):
        for timeStepIndex in range(len(trajectory) - 2):
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
            if self.drawPlanningAna:
                timeStep = trajectory[timeStepIndex]
                nextTimeStep = trajectory[timeStepIndex + 1]
                self.drawPlanningAna(timeStep, nextTimeStep)
            if self.interpolateState and timeStepIndex!= len(trajectory) - 1:
                statesToDraw = self.interpolateState(state, nextState)
            else:
                statesToDraw  = [state]
            for state in statesToDraw:
                screen = self.drawState(state, posterior)
        return
