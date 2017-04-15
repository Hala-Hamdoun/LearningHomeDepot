from __future__ import print_function

import numpy as np
import tensorflow as tf

def loadData(file):
    csvFile=open(file)
    lines=csvFile.read().splitlines()
    csvFile.close()
    splitLines=[]
    for line in lines:
        splitLines+=[line.split(',')]
    
    return splitLines;

print(loadData('../data/TimeSeriesPredictionTrain.csv'));
