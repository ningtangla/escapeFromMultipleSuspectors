import pandas as pd
import numpy as np
import AnalyticGeometryFunctions as ag
import itertools as it

def updateHypothesisInformation(hypothesisInformation,precisionHypothesisDF,decayHypothesisDF):
    hypothesisInformation['perceptionPrecision'] = precisionHypothesisDF.values
    hypothesisInformation['memoryDecay'] = decayHypothesisDF.values
    return hypothesisInformation


class ResetBeliefAndAttention():
    def __init__(self, sheepId, suspectorIds, possibleSubtleties, attentionLimitation, transferMultiAgentStatesToPositionDF, attention):
        self.sheepId = sheepId
        self.suspectorIds = suspectorIds
        self.possibleSubtleties = possibleSubtleties
        self.attentionLimitation = attentionLimitation
        self.transferMultiAgentStatesToPositionDF = transferMultiAgentStatesToPositionDF
        self.attention = attention

    def __call__(self, initialPhysicalState):
        identityListOfTuple = list(it.product(self.suspectorIds, [self.sheepId]))
        numberPairs = len(identityListOfTuple)
        numberSubtlety = len(self.possibleSubtleties)
        subtletyList=self.possibleSubtleties*numberPairs
        subtletyList.sort()
        identityListOfTuple=identityListOfTuple*numberSubtlety
        hypothesisLevel=[identityListOfTuple[i]+tuple([subtletyList[i]]) for i in range(numberPairs*numberSubtlety)]
        name=['wolfIdentity','sheepIdentity','chasingPrecision']
        priorIndex=pd.MultiIndex.from_tuples(hypothesisLevel,names=name)
        p=[np.log(1.0/len(priorIndex))]*len(priorIndex)
        initialHypothesisInformation=pd.DataFrame(p,priorIndex,columns=['logP'])
        initialHypothesisInformation['logPAttentionPrior']=initialHypothesisInformation['logP'].values
        initialHypothesisInformation['distanceProb']=[1.0] * len(priorIndex)
        allPairs = initialHypothesisInformation.groupby(['wolfIdentity','sheepIdentity']).mean().index
        attentionStatusForPair=np.random.multinomial(self.attentionLimitation,[1/len(allPairs)]*len(allPairs))
        attentionStatusForHypothesis=list(attentionStatusForPair)*numberSubtlety
        initialHypothesisInformation['attentionStatus']=attentionStatusForHypothesis
        initialHypothesisInformation['perceptionPrecision']=np.array([1]*len(priorIndex))
        initialHypothesisInformation['memoryDecay']=np.array([1]*len(priorIndex))

        attentionStatusDF = initialHypothesisInformation['attentionStatus']
        [precisionHypothesisDF,decayHypothesisDF]=self.attention(attentionStatusDF)
        initialHypothesisInformation = updateHypothesisInformation(initialHypothesisInformation, precisionHypothesisDF, decayHypothesisDF)

        initialAgentStates, initialAgentActions, timeStep, initialWolfIdAndSubtlety = initialPhysicalState
        initialPositionOldTimeDF = self.transferMultiAgentStatesToPositionDF(initialAgentStates)
        initialBeliefAndAttention = [initialHypothesisInformation, initialPositionOldTimeDF]
        return initialBeliefAndAttention

def computeObserveDF(hypothesisInformation,positionOldTimeDF,positionCurrentTimeDF):
    hypothesis = hypothesisInformation.index
    observeDF = pd.DataFrame(index=hypothesis,columns=['wolfDeviation'])
    wolfObjNums = hypothesis.get_level_values('wolfIdentity')
    sheepObjNums = hypothesis.get_level_values('sheepIdentity')
    wolfLocBefore = positionOldTimeDF.loc[wolfObjNums]
    sheepLocBefore = positionOldTimeDF.loc[sheepObjNums]
    wolfLocNow = positionCurrentTimeDF.loc[wolfObjNums]
    sheepLocNow = positionCurrentTimeDF.loc[sheepObjNums]
    wolfMotion = wolfLocNow - wolfLocBefore
    sheepMotion = sheepLocNow - sheepLocBefore
    seekingOrAvoidMotion = sheepLocBefore.values - wolfLocBefore.values
    distanceBetweenWolfAndSheep = np.sqrt(np.sum(np.power(wolfLocNow.values - sheepLocNow.values, 2), axis = 1))
    chasingAngle = ag.computeAngleBetweenVectors(wolfMotion, seekingOrAvoidMotion)
    escapingAngle = ag.computeAngleBetweenVectors(sheepMotion, seekingOrAvoidMotion)
    deviationAngleForWolf = np.random.vonmises(0,hypothesisInformation['perceptionPrecision'].values)
    deviationAngleForSheep = np.random.vonmises(0,hypothesisInformation['perceptionPrecision'].values)
    observeDF['wolfDeviation']=pd.DataFrame(chasingAngle.values+deviationAngleForWolf,index=hypothesis,columns=['wolfDeviation'])
    observeDF['sheepDeviation']=pd.DataFrame(escapingAngle.values+deviationAngleForSheep,index=hypothesis,columns=['sheepDeviation'])
    observeDF['distanceBetweenWolfAndSheep']=pd.DataFrame(distanceBetweenWolfAndSheep,index=hypothesis,columns=['distanceBetweenWolfAndSheep'])
    return observeDF

class TransferMultiAgentStatesToPositionDF():
    def __init__(self, numAgent):
        self.agentIds = list(range(numAgent))
        self.DFIndex = pd.Index(self.agentIds, name = 'Identity')
        self.DFColumns = pd.Index(['x', 'y'], name = 'Coordinate')

    def __call__(self, agentStates):
        positionDF = pd.DataFrame(np.array([agentStates[agentId] for agentId in self.agentIds]), index = self.DFIndex, columns = self.DFColumns)
        return positionDF

class UpdateBeliefAndAttentionState():
    def __init__(self,attention,computePosterior,attentionSwitch,transferMultiAgentStatesToPositionDF, attentionSwitchFrequency, beliefUpdateFrequency, burnTime):
        self.attention = attention
        self.computePosterior=computePosterior
        self.attentionSwitch = attentionSwitch
        self.transferMultiAgentStatesToPositionDF = transferMultiAgentStatesToPositionDF
        self.attentionSwitchFrequency=attentionSwitchFrequency
        self.beliefUpdateFrequency = beliefUpdateFrequency
        self.burnTime = burnTime

    def __call__(self, oldBeliefAndAttention, physicalState):
        agentStates, agentActions, timeStep, wolfIdAndSubtlety = physicalState
        hypothesisInformation, positionOldTimeDF = oldBeliefAndAttention
        if (timeStep % self.beliefUpdateFrequency == 0) and (timeStep > self.burnTime):
            positionCurrentTimeDF = self.transferMultiAgentStatesToPositionDF(agentStates)
            observeDF = computeObserveDF(hypothesisInformation, positionOldTimeDF, positionCurrentTimeDF)
            posteriorHypothesisDF = self.computePosterior(hypothesisInformation,observeDF)
            hypothesisInformation = posteriorHypothesisDF.copy()
            positionOldTimeDF = positionCurrentTimeDF.copy()
        if (timeStep % self.attentionSwitchFrequency == 0) and (timeStep > self.burnTime):
            hypothesisInformation = self.attentionSwitch(hypothesisInformation, positionOldTimeDF)
            attentionStatusDF = hypothesisInformation['attentionStatus']
            [precisionHypothesisDF,decayHypothesisDF]=self.attention(attentionStatusDF)
            hypothesisInformation = updateHypothesisInformation(hypothesisInformation, precisionHypothesisDF, decayHypothesisDF)
        newBeliefAndAttention = [hypothesisInformation, positionOldTimeDF]
        return newBeliefAndAttention

class UpdatePhysicalStateImagedByBelief():
    def __init__(self, updateFrequency, softParaForIdentity, softParaForSubtlety, lastIdentity = None, lastSubtlety = None):
        self.updateFrequency = updateFrequency
        self.softParaForIdentity = softParaForIdentity
        self.softParaForSubtlety = softParaForSubtlety
        self.lastIdentity = lastIdentity
        self.lastSubtlety = lastSubtlety
    def __call__(self, state):
        physicalState, beliefAndAttention = state
        agentStates, agentActions, timeStep, wolfIdAndSubtlety = physicalState
        if (timeStep % self.updateFrequency == 0) or (timeStep <= 1):

            hypothesisInformation, positionOldTimeDF = beliefAndAttention
            #attentedHypothesis = hypothesisInformation['attentionStatus']
            #attentedIdentity = attentedHypothesis.groupby(['wolfIdentity', 'sheepIdentity']).mean()
            #posteriorOnIdentitySlot = attentedIdentity.values / np.sum(attentedIdentity.values)
            probabilityOnHypothesisIdentity = np.exp(hypothesisInformation['logPAttentionPrior'])
            probabilityOnIdentitySlotByGroupbySum = probabilityOnHypothesisIdentity.groupby(['wolfIdentity','sheepIdentity']).sum()
            posteriorOnIdentitySlot = probabilityOnIdentitySlotByGroupbySum.values/np.sum(probabilityOnIdentitySlotByGroupbySum.values)
            softenPosteriorIdentityUnormalized = [np.power(prob, self.softParaForIdentity) for prob in posteriorOnIdentitySlot]
            softenPosteriorIdentity = np.array(softenPosteriorIdentityUnormalized) / np.sum(softenPosteriorIdentityUnormalized)

            sampledIdentityIndex = list(np.random.multinomial(1, softenPosteriorIdentity)).index(1)
            beliefWolfId, beliefSheepId = probabilityOnIdentitySlotByGroupbySum.index[sampledIdentityIndex]
            #maxIndex = np.argwhere(posteriorOnIdentitySlot == np.max(posteriorOnIdentitySlot)).flatten()
            #beliefWolfId = np.random.choice(maxIndex) + 1
            #beliefWolfId = list(np.random.multinomial(1, softenPosteriorIdentity)).index(1) + 1
            numOtherCondtionBeyondPair = hypothesisInformation.groupby(['wolfIdentity','sheepIdentity']).size().values[0]
            hypothesisInformation['identityProb'] = np.array(list(posteriorOnIdentitySlot) * numOtherCondtionBeyondPair)
            hypothesisInformation['identitySoftenProb'] = np.array(list(softenPosteriorIdentity) * numOtherCondtionBeyondPair)

            subtletyInformation = hypothesisInformation[hypothesisInformation.index.get_level_values('wolfIdentity') == beliefWolfId]['logP']
            originSubtletyProbs = np.exp(subtletyInformation.values)
            posteriorOnSubtlety = originSubtletyProbs / np.sum(originSubtletyProbs)
            softenPosteriorSubtletyUnormalized = [np.power(prob, self.softParaForSubtlety) for prob in posteriorOnSubtlety]
            softenPosteriorSubtlety = np.array(softenPosteriorSubtletyUnormalized) / np.sum(softenPosteriorSubtletyUnormalized)
            sampledSubtletyIndex = list(np.random.multinomial(1, softenPosteriorSubtlety)).index(1)
            beliefWolfId, beliefSheepId, beliefWolfSubtlety = subtletyInformation.index[sampledSubtletyIndex]
            #beliefWolfSubtlety = np.random.choice([0.001, 0.31, 0.92, 1.83, 3.3, 11.0, 500.0], p = softenPosteriorSubtlety)

            wolfIdAndSubtlety = [int(beliefWolfId), beliefWolfSubtlety]
            updatedPhysicalState = [agentStates, agentActions, timeStep, wolfIdAndSubtlety]
            state = [updatedPhysicalState, beliefAndAttention]
            self.lastIdentity = beliefWolfId
            self.lastSubtlety = beliefWolfSubtlety
        else:
            wolfIdAndSubtlety = [int(self.lastIdentity), self.lastSubtlety]
            updatedPhysicalState = [agentStates, agentActions, timeStep, wolfIdAndSubtlety]
            state = [updatedPhysicalState, beliefAndAttention]

        return state

if __name__=='__main__':
    print('end')
