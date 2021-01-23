import numpy as np
import itertools as it
import copy
import anytree
from anytree import AnyNode as Node
from anytree import RenderTree, findall
import math

class InitializeChildren:
    def __init__(self, actionSpace, transition, getActionPrior):
        self.actionSpace = actionSpace
        self.transition = transition
        self.getActionPrior = getActionPrior

    def __call__(self, node):
        state = list(node.id.values())[0]
        initActionPrior = self.getActionPrior(state)

        for action in self.actionSpace:
            nextState = self.transition(state, action)
            actionNode = Node(parent=node, id={action: action}, numVisited=0, sumValue=0,actionPrior=initActionPrior[action])

        return node

class Expand:
    def __init__(self, isTerminal, initializeChildren):
        self.isTerminal = isTerminal
        self.initializeChildren = initializeChildren

    def __call__(self, leafNode):
        currentState = list(leafNode.id.values())[0]
        if not self.isTerminal(currentState):
            leafNode.isExpanded = True
            leafNode = self.initializeChildren(leafNode)

        return leafNode

class ScoreChild:
    def __init__(self, cInit, cBase):
        self.cInit = cInit
        self.cBase = cBase
    def __call__(self, stateNode, actionNode):
        stateActionVisitCount = actionNode.numVisited
        stateVisitCount = stateNode.numVisited
        actionPrior = actionNode.actionPrior
        if actionNode.numVisited == 0:
            qScore = 0
        else:
            nextStateValues = [nextState.sumValue for nextState in actionNode.children]
            qScore = sum(nextStateValues) / stateActionVisitCount
        explorationRate = np.log((1 + stateVisitCount + self.cBase) / self.cBase) + self.cInit
        uScore = explorationRate * actionPrior * np.sqrt(stateVisitCount) / float(1 + stateActionVisitCount)#selfVisitCount is stateACtionVisitCount
        score = qScore + uScore
        return score

class SelectAction:
    def __init__(self, calculateScore):
        self.calculateScore = calculateScore

    def __call__(self, stateNode):
        scores = [self.calculateScore(stateNode, actionNode) for actionNode in list(stateNode.children)]
        maxIndex = np.argwhere(scores == np.max(scores)).flatten()
        selectedChildIndex = np.random.choice(maxIndex)
        selectedAction = stateNode.children[selectedChildIndex]
        return selectedAction

class PWidening:
    def __init__(self, alpha, C):
        self.alpha = alpha
        self.C = C

    def __call__(self, stateNode, actionNode):
        numActionVisit = actionNode.numVisited
        #print(numActionVisit)
        if numActionVisit == 0:
            return True
        else:
            k = math.ceil(self.C*pow(numActionVisit, self.alpha))
            return (k> len(actionNode.children))

class ExpandNextState:
    def __init__(self, transitionFunction, pWidening):
        self.transitionFunction = transitionFunction
        self.pWidening = pWidening

    def __call__(self, stateNode, actionNode):
        if self.pWidening(stateNode, actionNode):
            state = list(stateNode.id.values())[0]
            action = list(actionNode.id.values())[0]
            nextState = self.transitionFunction(state, action)
            nextStateNode = Node(parent=actionNode, id={action: nextState}, numVisited=0, sumValue=0,
                     isExpanded=False)

        return actionNode.children


class SelectNextState:
    def __init__(self, selectAction):
        self.selectAction = selectAction

    def __call__(self, stateNode, actionNode):
        nextPossibleState = actionNode.children
        if actionNode.numVisited == 0:
            probNextStateVisits = [1/len(nextPossibleState) for nextState in nextPossibleState]
            nextState = np.random.choice(nextPossibleState, 1, p = probNextStateVisits)
        else:
            for child in actionNode.children:
                if child.numVisited == 0:
                    return child
                else:
                    probNextStateVisits = [nextState.numVisited/actionNode.numVisited for nextState in actionNode.children]
                    nextState = np.random.choice(nextPossibleState, 1, p = probNextStateVisits)
        return nextState[0]


class RollOut:
    def __init__(self, rolloutPolicy, maxRolloutStep, transitionFunction, rewardFunction, isTerminal, rolloutHeuristic):
        self.transitionFunction = transitionFunction
        self.rewardFunction = rewardFunction
        self.maxRolloutStep = maxRolloutStep
        self.rolloutPolicy = rolloutPolicy
        self.isTerminal = isTerminal
        self.rolloutHeuristic = rolloutHeuristic

    def __call__(self, leafNode):
        currentState = list(leafNode.id.values())[0]
        totalRewardForRollout = 0

        for rolloutStep in range(self.maxRolloutStep):
            action = self.rolloutPolicy(currentState)
            nextState = self.transitionFunction(currentState, action)
            totalRewardForRollout += self.rewardFunction(currentState, action, nextState)
            if self.isTerminal(currentState):
                break

            currentState = nextState

        heuristicReward = 0
        if not self.isTerminal(currentState):
            heuristicReward = self.rolloutHeuristic(currentState)
        totalRewardForRollout += heuristicReward

        return totalRewardForRollout

def backup(value, nodeList): #anytree lib
    for node in nodeList:
        node.sumValue += value
        node.numVisited += 1

class OutputAction():
    def __init__(self, numActionPlaned, actionSpace):
        self.numActionPlaned = numActionPlaned
        self.actionSpace = actionSpace
        self.numActionSpace = len(actionSpace)
        self.actionIndexCombosForActionPlaned = list(it.product(range(self.numActionSpace), repeat = self.numActionPlaned))
        self.numActionIndexCombos = len(self.actionIndexCombosForActionPlaned)
    def __call__(self, roots):
        grandchildrenVisit = np.sum([[child.numVisited for child in findall(root, lambda node: node.depth == self.numActionPlaned)] for root in roots], axis=0)
        maxIndex = np.argwhere(grandchildrenVisit == np.max(grandchildrenVisit)).flatten()
        selectedActionIndexCombos = np.random.choice(maxIndex)
        action = [self.actionSpace[actionIndex] for actionIndex in self.actionIndexCombosForActionPlaned[selectedActionIndexCombos]]
        return action

class PWMultipleTrees:
    def __init__(self, numSimulation, selectAction, selectNextState, expand, expandNewState, estimateValue, backup, outputAction):
        self.numSimulation = numSimulation
        self.selectAction = selectAction
        self.selectNextState = selectNextState
        self.expand = expand
        self.expandNewState = expandNewState
        self.estimateValue = estimateValue
        self.backup = backup
        self.outputAction = outputAction

    def __call__(self, currRoots):
        numTree = len(currRoots)
        roots = []

        for treeIndex in range(numTree):
            currTreeRoot = copy.deepcopy(currRoots[treeIndex])
            currTreeRoot = self.expand(currTreeRoot)
            for exploreStep in range(self.numSimulation):
                currentNode = currTreeRoot
                nodePath = [currentNode]

                while currentNode.isExpanded:
                    actionNode = self.selectAction(currentNode)
                    allNextStateNodes = self.expandNewState(currentNode, actionNode)
                    #print(allNextStateNodes)
                    nextStateNode = self.selectNextState(currentNode, actionNode)
                    #print(nextStateNode)

                    nodePath.append(actionNode)
                    nodePath.append(nextStateNode)
                    currentNode = nextStateNode

                leafNode = self.expand(currentNode)
                value = self.estimateValue(leafNode)
                self.backup(value, nodePath)

            roots.append(currTreeRoot)
        action = self.outputAction(roots)
        return action

def establishPlainActionDist(root):
    visits = np.array([child.numVisited for child in root.children])
    actionProbs = visits / np.sum(visits)
    actions = [list(child.id.keys())[0] for child in root.children]
    actionDist = dict(zip(actions, actionProbs))
    return actionDist


def establishSoftmaxActionDist(root):
    visits = np.array([child.numVisited for child in root.children])
    expVisits = np.exp(visits)
    actionProbs = expVisits / np.sum(expVisits)
    actions = [list(child.id.keys())[0] for child in root.children]
    actionDist = dict(zip(actions, actionProbs))
    return actionDist
