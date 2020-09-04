import pandas as pd
import numpy as np
import scipy.stats as stats
import math

def calAngleLikelihoodLogModifiedForPiRange(angle, kappa):
    return stats.vonmises.logpdf(angle, kappa) + np.log(2)

class CalPosteriorLog():
    def __init__(self, minDistance):
        self.minDistance = minDistance
    def __call__(self, hypothesesInformation, observedData):    
        hypothesesInformation['chasingLikelihoodLog'] = calAngleLikelihoodLogModifiedForPiRange(observedData['wolfDeviation'], 1/(1/hypothesesInformation.index.get_level_values('chasingPrecision') + 1/hypothesesInformation['perceptionPrecision']))
        hypothesesInformation['escapingLikelihoodLog'] = 0
        #originPrior = np.exp(hypothesesInformation['logP'].values)
        #normalizedPrior = np.maximum([1e-20] * len(originPrior), originPrior / np.sum(originPrior))  
        #hypothesesInformation['beforeLogPAfterDecay'] = np.log(normalizedPrior) * hypothesesInformation['memoryDecay']
        hypothesesInformation['beforeLogPAfterDecay'] = hypothesesInformation['memoryDecay'] * hypothesesInformation['logP']
        #print(np.exp(hypothesesInformation['logP']).values)
        #print('***', originPrior)
        #print('!!!', normalizedPrior)
        #distanceLikelihoodLog = np.array([-50 if distance <= self.minDistance else 0 for distance in observedData['distanceBetweenWolfAndSheep'].values])
        distanceLikelihoodLog = 0
        hypothesesInformation['logP'] = hypothesesInformation['beforeLogPAfterDecay'] + hypothesesInformation['chasingLikelihoodLog'] \
                                        + hypothesesInformation['escapingLikelihoodLog'] + distanceLikelihoodLog
        #originLogP = hypothesesInformation['logP'].values
        #sizeHypothesesSpace = len(originLogP)
        #logPWithUpLowBound = np.minimum([600] * sizeHypothesesSpace, np.maximum([-600] * sizeHypothesesSpace, originLogP))
        #hypothesesInformation['logP'] = logPWithUpLowBound
        #print('***', hypothesesInformation['logP'].values)
        return hypothesesInformation
    



 
