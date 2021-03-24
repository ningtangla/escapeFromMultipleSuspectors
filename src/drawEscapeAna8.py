from collections import OrderedDict
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import os

import trajectoriesSaveLoad as tsl

class Readcsv:
    def __init__(self, getCSVSavePathByCondition, columnNames):
        self.getCSVSavePathByCondition = getCSVSavePathByCondition
        self.columnNames = columnNames
    def __call__(self, condition):
        getCSVSavePath = self.getCSVSavePathByCondition(tsl.readParametersFromDf(condition))
        CSVSavePath = getCSVSavePath({})
        results = pd.read_csv(CSVSavePath, header = None, skiprows = [0], names = self.columnNames)#, header = None)
        mean = results.mean()
        return mean

def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['alpha'] = [0.25]
    #manipulatedVariables['attType'] = ['idealObserver']#, 'hybrid4']
    manipulatedVariables['attType'] = ['hybrid4']#, 'preAttention']
    #manipulatedVariables['attType'] = ['preAttention']
    #manipulatedVariables['attType'] = ['idealObserver', 'preAttention', 'attention4', 'hybrid4']
    #manipulatedVariables['attType'] = ['preAttentionMem0.65', 'preAttentionMem0.25', 'preAttentionPre0.5', 'preAttentionPre4.5']
    manipulatedVariables['C'] = [2]
    manipulatedVariables['minAttDist'] = [10.0]
    manipulatedVariables['rangeAtt'] = [20.0]
    manipulatedVariables['cBase'] = [50]
    manipulatedVariables['numTrees'] = [4]
    manipulatedVariables['numSim'] = [184]
    #manipulatedVariables['actRatio'] = [1.0]
    #manipulatedVariables['burnTime'] = [0]
    manipulatedVariables['softId'] = [0.1, 1.0, 10.0, 100.0]
    manipulatedVariables['softSubtlety'] = [1.0]
    manipulatedVariables['actCost'] = [0]
    manipulatedVariables['aliveBouns'] = [0.1]
    manipulatedVariables['damp'] = [1.0]
    manipulatedVariables['measure'] = ['escape']

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', 'data', 'mcts',
                                'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    measurementEscapeExtension = '.csv'
    getCSVSavePathByCondition = lambda condition: tsl.GetSavePath(trajectoryDirectory, measurementEscapeExtension, condition)
    #columnNames = [500.0, 11.0, 3.3, 1.83, 0.92, 0.31, 0.001]
    #columnNames = [500.0, 3.3, 0.92, 0.01]
    columnNames = [500.0, 3.3, 0.92, 0.01]
    #columnNames = [500.0]
    readcsv = Readcsv(getCSVSavePathByCondition, columnNames)

    precisionToSubtletyDict={500.0:0, 50.0:5, 11.0:30, 3.3:60, 1.83:90, 0.92:120, 0.31:150, 0.01: 180}

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index = modelIndex)

    modelResultDf = toSplitFrame.groupby(levelNames).apply(readcsv)
    toDropLevels = ['alpha', 'C', 'cBase', 'numTrees', 'actCost', 'aliveBouns', 'damp', 'softSubtlety',
            'measure', 'attType', 'numSim']
    modelResultDf.index = modelResultDf.index.droplevel(toDropLevels)
    fig = plt.figure()
    numColumns = len(manipulatedVariables['minAttDist'])
    numRows = len(manipulatedVariables['rangeAtt'])
    plotCounter = 1
    for key, group in modelResultDf.groupby(['rangeAtt', 'minAttDist']):
        columnNamesAsSubtlety = [precisionToSubtletyDict[precision] for precision in group.columns]
        group.columns = columnNamesAsSubtlety
        group = group.stack()
        #print(group)
        group.index.names = ['minAttDist', 'rangeAtt', 'softId', 'chasingSubtlety']
        #print(group)
        group.index = group.index.droplevel(['minAttDist', 'rangeAtt'])
        group = group.to_frame()

        group.columns = ['model']
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        if (plotCounter) % max(numColumns, 2) == 1:
            axForDraw.set_ylabel(str(key[0]))

        if plotCounter <= numColumns:
            axForDraw.set_title(str(key[1]))
        for attentionType, grp in group.groupby('softId'):
            print(grp)
            grp.index = grp.index.droplevel('softId')
            #if str(attentionType) == manipulatedVariables['attType'][-1]:
            #    grp['human'] = [0.24, 0.51]
            #    grp['human'] = [0.6, 0.48, 0.37, 0.25, 0.24, 0.42, 0.51]
            #    grp.plot.line(ax = axForDraw, y = 'human', label = 'human', ylim = (0, 0.7), marker = 'o', rot = 0 )
            grp.plot.line(ax = axForDraw, y = 'model', label = str(attentionType), ylim = (0, 1.1), marker = 'o', rot = 0 )

        plotCounter = plotCounter + 1

    #plt.suptitle('Measurement = Perception Rate')
    #plt.suptitle('Measurement = Action Deviation')
    #plt.suptitle('Measurement = Velocity Diff')
    #plt.suptitle('Measurement = Escape rate')
    plt.suptitle('Hybrid4')
    #fig.text(x = 0.5, y = 0.92, s = 'Action Cost', ha = 'center', va = 'center')
    #fig.text(x = 0.5, y = 0.92, s = 'Min Attention Distance', ha = 'center', va = 'center')
    #fig.text(x = 0.05, y = 0.5, s = 'Attention Range', ha = 'center', va = 'center', rotation=90)
    #fig.text(x = 0.05, y = 0.5, s = 'Action Ratio', ha = 'center', va = 'center', rotation=90)
    plt.show()

if __name__ == "__main__":
    main()

