from __future__ import print_function

import numpy as np
import tensorflow as tf

def loadData(file):
    csvFile=open(file)
    lines=csvFile.read().splitlines()
    csvFile.close()
    products = [];
   
    for product in lines:
        products.append(product.split(","));
        
    return products[0][0];

def printRow(file, row):
    csvFile=open(file)
    lines=csvFile.read().splitlines()
    csvFile.close()
    products = [];
   
    for product in lines:
        products.append(product.split(","));
        
    return_row = [];

    for i in range( 0, len( products[row] ) ):
        return_row.append( products[row][i] );

    return return_row

print( printRow('./TimeSeriesPredictionTrain.csv', 0 ) );
