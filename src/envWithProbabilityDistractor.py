import os
import numpy as np
import pandas as pd
import pygame as pg
import itertools as it
import random
import anytree
import AnalyticGeometryFunctions as ag
import math
#np.random.seed(123)

class TransitionFunction():
    def __init__(self, resetPhysicalState, resetBeliefAndAttention, updatePhysicalState, transiteStateWithoutActionChange, updateBeliefAndAttention, updatePhysicalStateByBelief):
        self.resetPhysicalState = resetPhysicalState
        self.resetBeliefAndAttention = resetBeliefAndAttention
        self.updatePhysicalState = updatePhysicalState
        self.transiteStateWithoutActionChange = transiteStateWithoutActionChange
        self.updateBeliefAndAttention = updateBeliefAndAttention
        self.updatePhysicalStateByBelief = updatePhysicalStateByBelief

    def __call__(self, oldState, action):
        if oldState is None:
            newPhysicalState = self.resetPhysicalState()
            newBeliefAndAttention = self.resetBeliefAndAttention(newPhysicalState)
            newState = [newPhysicalState, newBeliefAndAttention] 
        else:
            oldPhysicalState, oldBeliefAndAttention = oldState
            
            newPhysicalState = self.updatePhysicalState(oldPhysicalState, action)
            
            stateBeforeNoActionChangeTransition = [newPhysicalState, oldBeliefAndAttention]
            physicalStateAfterNoActionChangeTransition, beliefAndAttentionAfterNoActionChangeTransition = self.transiteStateWithoutActionChange(stateBeforeNoActionChangeTransition) 
            newBeliefAndAttention = self.updateBeliefAndAttention(oldBeliefAndAttention, physicalStateAfterNoActionChangeTransition)

            newState = [physicalStateAfterNoActionChangeTransition, newBeliefAndAttention]
            newState = self.updatePhysicalStateByBelief(newState)
        return newState

class TransiteStateWithoutActionChange():
    def __init__(self, maxFrame, isTerminal, transiteMultiAgentMotion, render, renderOn):
        self.maxFrame = maxFrame
        self.isTerminal = isTerminal
        self.transiteMultiAgentMotion = transiteMultiAgentMotion
        self.render = render
        self.renderOn = renderOn
    def __call__(self, state):
        for frame in range(self.maxFrame):
            physicalState, beliefAndAttention = state 
            agentStates, agentActions, timeStep, wolfIdAndSubtlety = physicalState
            change = np.random.randint(0, self.maxFrame, len(agentStates))
            changeLabel = 1 * (change == 0)
            changeLabel[0] = 0
            changeLabel[wolfIdAndSubtlety[0]] = 0
            currentActionsPolar = np.array([ag.transiteCartesianToPolar(action) for action in agentActions])
            polarAfterChange = np.random.uniform(-math.pi*1/3, math.pi*1/3) * np.array(changeLabel) + currentActionsPolar
            actionsAfterChange = np.array([ag.transitePolarToCartesian(polar) for polar in polarAfterChange]) * np.linalg.norm(agentActions[1])
            if self.renderOn == True:
                self.render(state)
            if self.isTerminal(state):
                break
            newAgentStates, newAgentActions = self.transiteMultiAgentMotion(agentStates, actionsAfterChange) 
            newPhysicalState = [newAgentStates, newAgentActions, timeStep, wolfIdAndSubtlety]
            stateAfterNoActionChangeTransition = [newPhysicalState, beliefAndAttention]
            state = stateAfterNoActionChangeTransition
        return state

class IsTerminal():
    def __init__(self, sheepId, minDistance):
        self.sheepId = sheepId
        self.minDistance = minDistance

    def __call__(self, state):
        terminal = False
        physicalState, beliefAndAttention = state
        agentStates, agentActions, timeStep, wolfIdAndSubtlety = physicalState
        wolfId, wolfSubtlety = wolfIdAndSubtlety
        sheepPosition = agentStates[self.sheepId]
        wolfPosition = agentStates[wolfId]
        if np.sum(np.power(sheepPosition - wolfPosition, 2)) ** 0.5 <= self.minDistance:
            terminal = True
        return terminal   


class Render():
    def __init__(self, numAgent, screen, surfaceWidth, surfaceHeight, screenColor, sheepColor, wolfColor, circleSize, saveImage, saveImageFile, isTerminal):
        self.numAgent = numAgent
        self.screen = screen
        self.surfaceWidth = surfaceWidth
        self.surfaceHeight = surfaceHeight
        self.screenColor = screenColor
        self.sheepColor = sheepColor
        self.wolfColor = wolfColor
        self.circleSize = circleSize
        self.saveImage = saveImage
        self.saveImageFile = saveImageFile
        self.isTerminal = isTerminal

    def __call__(self, state):
        physicalState, beliefAndAttention = state 
        agentStates, agentActions, timeStep, wolfIdAndSubtlety = physicalState
        groundTruthWolf, groundTruthSubtlety = wolfIdAndSubtlety

        hypothesisInformation, positionOldTimeDF = beliefAndAttention
        posteriorAllHypothesesBeforeNormalization = np.exp(hypothesisInformation['logP'])
        posteriorAllHypotheses = posteriorAllHypothesesBeforeNormalization / (np.sum(posteriorAllHypothesesBeforeNormalization))
        posteriorAllWolf = posteriorAllHypotheses.groupby(['wolfIdentity']).sum().values
        
        attentionStatus = hypothesisInformation.groupby(['wolfIdentity'])['attentionStatus'].mean().values
        attentionSlot = np.concatenate(np.argwhere(attentionStatus != 0)) + 1
        beliefSurface = pg.Surface((self.surfaceWidth, self.surfaceHeight))

        wolfColors = [np.ones(3) * 250 * (1 - 1.5*wolfBelief) + self.wolfColor * 1.5*wolfBelief for wolfBelief in posteriorAllWolf]
        circleColorList = np.array([self.sheepColor] + wolfColors)
        #circleColorList[groundTruthWolf] = circleColorList[groundTruthWolf] + np.array([0, 0, 255])
        
        for j in range(1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit
            self.screen.fill(self.screenColor)
            for i in range(self.numAgent):
                oneAgentState = agentStates[i]
                oneAgentPosition = np.array(oneAgentState)
                if i in attentionSlot:
                    pg.draw.circle(self.screen, np.array([0, 0, 255]), [np.int(oneAgentPosition[0]),np.int(oneAgentPosition[1])], 
                            min(0, 5*int(attentionStatus[i - 1]) + 10), min(0, 5*int(attentionStatus[i - 1])))
                pg.draw.circle(self.screen, np.clip(circleColorList[i], 0, 255), [np.int(oneAgentPosition[0]),np.int(oneAgentPosition[1])], self.circleSize)
                if self.isTerminal(state) and i == groundTruthWolf:
                    pg.draw.circle(self.screen, np.array([255, 0, 0]), [np.int(oneAgentPosition[0]),np.int(oneAgentPosition[1])], int(self.circleSize*1.5))
            
            pg.display.flip()
            if self.saveImage==True:
                currentDir = os.getcwd()
                parentDir = os.path.abspath(os.path.join(currentDir, os.pardir))
                saveImageDir=parentDir+'/src/data/'+self.saveImageFile
                #if j == 1 :
                #    saveImageDir=parentDir+'/src/data/'+self.saveImageFile+'/groundtruth'
                if self.isTerminal(state):
                    for pauseTimeIndex in range(90):
                        filenameList = os.listdir(saveImageDir)
                        pg.image.save(self.screen,saveImageDir+'/'+str(len(filenameList))+'.png')
                        pg.time.wait(1)

                filenameList = os.listdir(saveImageDir)
                pg.image.save(self.screen,saveImageDir+'/'+str(len(filenameList))+'.png')
                pg.time.wait(1)

class MctsRender():
    def __init__(self, numAgent, screen, surfaceWidth, surfaceHeight, screenColor, sheepColor, wolfColor, distractorColor, circleSize, saveImage, saveImageFile):
        self.numAgent = numAgent
        self.screen = screen
        self.surfaceWidth = surfaceWidth
        self.surfaceHeight = surfaceHeight
        self.screenColor = screenColor
        self.sheepColor = sheepColor
        self.wolfColor = wolfColor
        self.distractorColor = distractorColor
        self.circleSize = circleSize
        self.saveImage = saveImage
        self.saveImageFile = saveImageFile
    def __call__(self, currNode, nextNode, roots, backgroundScreen):

        parentNumVisit = currNode.num_visited
        parentValueToTal = currNode.sum_value
        state = list(currNode.id.values())[0] 
        physicalState, beliefAndAttention = state 
        agentStates, agentActions, timeStep, wolfIdAndSubtlety = physicalState
        wolfId, wolfSubtlety = wolfIdAndSubtlety 
        hypothesisInformation, positionOldTimeDF = beliefAndAttention
        posteriorAllHypothesesBeforeNormalization = np.exp(hypothesisInformation['logP'])
        posteriorAllHypotheses = posteriorAllHypothesesBeforeNormalization / (np.sum(posteriorAllHypothesesBeforeNormalization))
        posteriorAllWolf = posteriorAllHypotheses.groupby(['wolfIdentity']).sum().values
            
         
        childNumVisit = nextNode.num_visited
        childValueToTal = nextNode.sum_value
        nextState = list(nextNode.id.values())[0]
        nextPhysicalState, nextBeliefAndAttention = nextState 
        nextAgentStates, nextAgentActions, nextTimeStep, nextWolfIdAndSubtlety = nextPhysicalState
        
        lineWidth = nextNode.num_visited + 1 
        if len(roots) > 0 and nextNode.depth == 1:
            nodeIndex = currNode.children.index(nextNode)
            grandchildren_visit = np.sum([[child.num_visited for child in anytree.findall(root, lambda node: node.depth == 1)] for root in roots], axis=0)
            lineWidth = lineWidth + grandchildren_visit[nodeIndex] 

        font = pg.font.SysFont("Arial", 12)

        surfaceToDraw = pg.Surface((self.surfaceWidth, self.surfaceHeight))
        surfaceToDraw.fill(self.screenColor)
        if backgroundScreen == None:
            backgroundScreen = pg.Surface((self.surfaceWidth, self.surfaceHeight))
            beliefSurface = pg.Surface((self.surfaceWidth, self.surfaceHeight))
            backgroundScreen.fill(self.screenColor)
            self.screen.fill(self.screenColor)
            
            wolfColors = [np.ones(3) * 250 * (1 - 1.5*wolfBelief) + 1.5*self.wolfColor * wolfBelief for wolfBelief in posteriorAllWolf]
            circleColorList = np.array([self.sheepColor] + wolfColors)
            
            attentionStatus = hypothesisInformation.groupby(['wolfIdentity'])['attentionStatus'].mean().values
            attentionSlot = np.concatenate(np.argwhere(attentionStatus != 0)) + 1
            for i in range(self.numAgent):
                oneAgentState = agentStates[i]
                oneAgentNextState = nextAgentStates[i]
                oneAgentPosition = np.array(oneAgentState)
                oneAgentNextPosition = np.array(oneAgentNextState)
                if i in attentionSlot:
                    pg.draw.circle(backgroundScreen, np.array([0, 0, 255]), [np.int(oneAgentPosition[0]),np.int(oneAgentPosition[1])], 5*int(attentionStatus[i - 1]) + 10,
                            5*int(attentionStatus[i - 1]))
                pg.draw.circle(backgroundScreen, np.clip(circleColorList[i],0,255), [np.int(oneAgentPosition[0]),np.int(oneAgentPosition[1])], self.circleSize)
            if self.saveImage==True:
                for i in range(1):
                    currentDir = os.getcwd()
                    parentDir = os.path.abspath(os.path.join(currentDir, os.pardir))
                    saveImageDir=parentDir+'/src/data/'+self.saveImageFile
                    filenameList = os.listdir(saveImageDir)
                    pg.image.save(backgroundScreen,saveImageDir+'/'+str(len(filenameList))+'.png')
         
        surfaceToDraw.set_alpha(180)
        surfaceToDraw.blit(backgroundScreen, (0,0))
        self.screen.blit(surfaceToDraw, (0, 0)) 
    
        pg.display.flip()
        pg.time.wait(1)
        
        
        circleColorList = [self.distractorColor] * self.numAgent
        circleColorList[wolfId] = self.wolfColor
        circleColorList[0] = self.sheepColor
        for j in range(1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit
                
            for i in range(self.numAgent):
                oneAgentState = agentStates[i]
                oneAgentNextState = nextAgentStates[i]
                oneAgentPosition = np.array(oneAgentState)
                oneAgentNextPosition = np.array(oneAgentNextState)
                if i == 0: 
                    pg.draw.line(surfaceToDraw, np.ones(3) * 240, [np.int(oneAgentPosition[0]), np.int(oneAgentPosition[1])], [np.int(oneAgentNextPosition[0]),np.int(oneAgentNextPosition[1])], lineWidth)
                    pg.draw.circle(surfaceToDraw, circleColorList[i], [np.int(oneAgentNextPosition[0]),np.int(oneAgentNextPosition[1])], self.circleSize)
                if i == wolfId:
                    pg.draw.circle(surfaceToDraw, circleColorList[i], [np.int(oneAgentPosition[0]),np.int(oneAgentPosition[1])], self.circleSize)
                    pg.draw.circle(surfaceToDraw, circleColorList[i], [np.int(oneAgentNextPosition[0]),np.int(oneAgentNextPosition[1])], self.circleSize)
             
            self.screen.blit(surfaceToDraw, (0, 0)) 
            pg.display.flip()
            pg.time.wait(1)
            backgroundScreenToReturn = self.screen.copy()
            
            if self.saveImage==True:
                currentDir = os.getcwd()
                parentDir = os.path.abspath(os.path.join(currentDir, os.pardir))
                saveImageDir=parentDir+'/src/data/'+self.saveImageFile
                filenameList = os.listdir(saveImageDir)
                pg.image.save(self.screen,saveImageDir+'/'+str(len(filenameList))+'.png')
        return self.screen

if __name__ == '__main__':
    a = TransitionFunction
    __import__('ipdb').set_trace()
