from __future__ import print_function

import numpy as np
import tensorflow as tf

def loadData(file):
    csvFile=open(file)
    lines=csvFile.read().splitlines()
    csvFile.close()
    products = [[] * len(lines)];
   
    for product in lines:
        products.push(product.split(","));
        
    return splitLines;

print(loadData('./TimeSeriesPredictionTrain.csv'));
