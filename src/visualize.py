import matplotlib.pyplot as plt
import pandas as pd


def makeTitle(xlabel, ylabel, graphIndex):
	return '%s%s vs %s' % (xlabel, graphIndex, ylabel)


def splitDictionary(originalDict, splitFactor):
	newDict = [{key: value[index] for key, value in originalDict.items()} for index in range(splitFactor)]
	return newDict


def dictToDataframe(data, axisName, lineVariableIndex):
	numOfDependentVariable = len(list(data.values())[0])
	splitedDependentVaraibles = splitDictionary(data, numOfDependentVariable)
	dataDFs = [pd.Series(dictionary).rename_axis(axisName).unstack(level=lineVariableIndex) for dictionary in splitedDependentVaraibles]
	return dataDFs


def drawPerGraph(dataDF, title):
	plt.title(title)
	dataDF.plot(title=title)
	plt.savefig(title)


def draw(data, independetVariablesName, lineVariableIndex=0, xVariableIndex=1):
	dataDFs = dictToDataframe(data, independetVariablesName, lineVariableIndex)
	plt.figure()
	titles = [makeTitle('escapeRate', independetVariablesName[xVariableIndex], graphIndex=index) for index in range(len(dataDFs))]
	[drawPerGraph(dataDF, title) for dataDF, title in zip(dataDFs, titles)]


if __name__ == '__main__':

    data = {(1, 0): [9/15], (1, 30): [5/15], (1, 60): [2/15], (1, 90): [1/15], (1, 120): [2/15], (1, 150): [6/15], (1, 180): [10/15],
            (2, 0): [12/15], (2, 30): [8/15], (2, 60): [7/15], (2, 90): [2/15], (2, 120): [4/15], (2, 150): [10/15], (2, 180): [13/15],
            (2.1, 0): [11/15], (2.1, 30): [2/15], (2.1, 60): [1/15], (2.1, 90): [0/15], (2.1, 120): [5/15], (2.1, 150): [10/15], (2.1, 180): [13/15],
            (2.2, 0): [9/15], (2.2, 30): [8/15], (2.2, 60): [7/15], (2.2, 90): [3/15], (2.2, 120): [4/15], (2.2, 150): [10/15], (2.2, 180): [14/15],
            (3, 0): [13/15], (3, 30): [13/15], (3, 60): [12/15], (3, 90): [10/15], (3, 120): [10/15], (3, 150): [12/15], (3, 180): [15/15],
            (4, 0): [0.6], (4, 30): [0.48], (4, 60): [0.37], (4, 90): [0.25], (4, 120): [0.24], (4, 150): [0.43], (4, 180): [0.5]}
    #data = {(128, 2): [10, 20], (128, 4): [10, 30]}
    draw(data, ['beliefModelType', 'chasingSubtlety'])
