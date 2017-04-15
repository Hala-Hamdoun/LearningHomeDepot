from __future__ import print_function

import numpy as np
import tensorflow as tf

def loadData(file):
    csvFile=open(file)
    lines=csvFile.read().splitlines()
    csvFile.close()
    products = [];
    productTriples = []
    for product in lines:
        products.append(product.split(","));
    
    #Insert 0 Dates here

    for product in products:
        for i in range(len(product)):
            date = product[i];
            product[i] = date.split("|");
            
    for productNumber in range(len(products)):
        product = products[productNumber];
        for i in range(len(product)-2):
            productTriples.append([product[i],product[i+1],product[i+2]]);
            
    return productTriples;

#print(loadData('./TimeSeriesPredictionTrain.csv'));


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

#print( printRow('./TimeSeriesPredictionTrain.csv', 0 ) );

def getFeatureResultFormat(productTriples):
    features = []; #[two days ago views][one day ago views]
    classifications = [] # [third day views];

    for triple in productTriples:
        if(len(triple) == 3):        
            twoAgo = triple[0][1]        
            oneAgo = triple[1][1];        
            features.append([twoAgo, oneAgo]);
            classifications.append([triple[2][1]]);

    return features, classifications;

#trips = loadData('./TimeSeriesPredictionTrain.csv');
#features, classes = getFeatureResultFormat(trips)
#print(trips[-1]);
#print("Features: " + str(features[-1]) + " class: " + str(classes[-1]));
        
    
    














