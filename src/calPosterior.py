import pandas as pd
import numpy as np
import scipy.stats as stats
import math

def calAngleLikelihoodLogModifiedForPiRangeAndMemoryDecay(angle, center):
    return stats.vonmises.logpdf(angle, center) + np.log(2) + np.log(math.pi)

class CalPosteriorLog():
    def __init__(self, minDistance):
        self.minDistance = minDistance
    def __call__(self, hypothesesInformation, observedData):    
        hypothesesInformation['chasingLikelihoodLog'] = calAngleLikelihoodLogModifiedForPiRangeAndMemoryDecay(observedData['wolfDeviation'], 1/(1/hypothesesInformation.index.get_level_values('chasingPrecision') + 1/hypothesesInformation['perceptionPrecision']))
        hypothesesInformation['escapingLikelihoodLog'] = 0
        hypothesesInformation['beforeLogPAfterDecay'] = hypothesesInformation['memoryDecay'] * hypothesesInformation['logP']
        #distanceLikelihoodLog = np.array([-50 if distance <= self.minDistance else 0 for distance in observedData['distanceBetweenWolfAndSheep'].values])
        distanceLikelihoodLog = 0
        hypothesesInformation['logP'] = hypothesesInformation['beforeLogPAfterDecay'] + hypothesesInformation['chasingLikelihoodLog'] \
                                        + hypothesesInformation['escapingLikelihoodLog'] + distanceLikelihoodLog
        return hypothesesInformation
    



 
