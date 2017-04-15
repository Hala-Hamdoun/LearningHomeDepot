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

def numDates(file):
    csvFile=open(file)
    lines=csvFile.read().splitlines()
    csvFile.close()
    products = [];

    for product in lines:
        products.append(product.split(","));
        
    return_row = [];

    for i in range( 0, len( products ) ):
        return_row.append( len(products[i]) );

    return return_row

def dateParser



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

def splitTestTrain(features, classes, breakPercent):
    if len(features) != len(classes):
        print("BIG PROBLEM, YUUGE");
        return;
    
    breakPoint = int(len(features) * breakPercent);
    return [features[0:breakPoint], classes[0:breakPoint]],[features[breakPoint:],classes[breakPoint:]];

    

trips = loadData('./TimeSeriesPredictionTrain.csv');
features, classes = getFeatureResultFormat(trips)
train, test = splitTestTrain(features, classes, .8);
xTr, yTr = train;
xTe, yTe = test;
        
    
    






print( numDates( './TimeSeriesPredictionTrain.csv' ) );
#print( printRow('./TimeSeriesPredictionTrain.csv', 0 ) );
