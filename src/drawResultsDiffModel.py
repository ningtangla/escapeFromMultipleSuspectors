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
    manipulatedVariables['alphaForStateWidening'] = [0.25]
    #manipulatedVariables['attentionType'] = ['idealObserver']
    #manipulatedVariables['attentionType'] = ['hybrid4']
    manipulatedVariables['attentionType'] = ['preAttention', 'attention4', 'hybrid4', 'idealObserver']#, 'attention3', 'hybrid3']
    manipulatedVariables['CForStateWidening'] = [2]
    #manipulatedVariables['minAttentionDistance'] = [8.5, 12.5]#[18.0, 40.0]
    manipulatedVariables['minAttentionDistance'] = [20.0, 40.0]
    manipulatedVariables['rangeAttention'] = [4]
    manipulatedVariables['cBase'] = [50]
    manipulatedVariables['numTrees'] = [2]
    manipulatedVariables['numSimulationTimes'] = [72]
    manipulatedVariables['actionRatio'] = [0.2]
    manipulatedVariables['burnTime'] = [4]
 
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', 'data', 'mcts',
                                'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    measurementEscapeExtension = '.csv'
    getCSVSavePathByCondition = lambda condition: tsl.GetSavePath(trajectoryDirectory, measurementEscapeExtension, condition)
    columnNames = [500.0, 11.0, 3.3, 1.83, 0.92, 0.31, 0.001]
    readcsv = Readcsv(getCSVSavePathByCondition, columnNames)

    precisionToSubtletyDict={500.0:0, 50.0:5, 11.0:30, 3.3:60, 1.83:90, 0.92:120, 0.31:150, 0.001: 180}
    
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index = modelIndex)

    modelResultDf = toSplitFrame.groupby(levelNames).apply(readcsv)
    toDropLevels = ['alphaForStateWidening', 'CForStateWidening', 'cBase', 'numTrees', 'numSimulationTimes', 'actionRatio', 'burnTime']
    modelResultDf.index = modelResultDf.index.droplevel(toDropLevels)
    fig = plt.figure()
    numColumns = len(manipulatedVariables['minAttentionDistance'])
    numRows = len(manipulatedVariables['rangeAttention'])
    plotCounter = 1
    for key, group in modelResultDf.groupby(['rangeAttention', 'minAttentionDistance']):
        columnNamesAsSubtlety = [precisionToSubtletyDict[precision] for precision in group.columns]
        group.columns = columnNamesAsSubtlety
        group = group.stack()
        group.index.names = ['attentionType', 'minAttentionDistance', 'rangeAttention', 'chasingSubtlety']
        group.index = group.index.droplevel(['minAttentionDistance', 'rangeAttention'])
        group = group.to_frame()
        
        group.columns = ['model']
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        if (plotCounter) % max(numColumns, 2) == 1:
            axForDraw.set_ylabel(str(key[0]))
        
        if plotCounter <= numColumns:
            axForDraw.set_title(str(key[1]))
        for attentionType, grp in group.groupby('attentionType'):
            grp.index = grp.index.droplevel('attentionType')
            if str(attentionType) == manipulatedVariables['attentionType'][-1]:
                grp['human'] = [0.6, 0.48, 0.37, 0.25, 0.24, 0.42, 0.51]
                grp.plot.line(ax = axForDraw, y = 'human', label = 'human', ylim = (0, 1.01), marker = 'o', rot = 0 )
            grp.plot.line(ax = axForDraw, y = 'model', label = str(attentionType), ylim = (0, 1.01), marker = 'o', rot = 0 )
       
        plotCounter = plotCounter + 1

    plt.suptitle('Measurement = Escape Rate of Hybrid')
    fig.text(x = 0.5, y = 0.92, s = 'Min Attention Distance', ha = 'center', va = 'center')
    fig.text(x = 0.05, y = 0.5, s = 'Attention Range', ha = 'center', va = 'center', rotation=90)
    plt.show()

if __name__ == "__main__":
    main()

