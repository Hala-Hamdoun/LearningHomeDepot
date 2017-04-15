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
        productTriples.append([]);
    
    #Insert 0 Dates here

    for product in products:
        for i in range(len(product)):
            date = product[i];
            product[i] = date.split("|");
            

    for productNumber in range(len(productTriples)):
        product = products[productNumber];
        for i in range(len(product)-2):
            productTriples[productNumber].append([product[i],product[i+1],product[i+2]]);
            

    return productTriples[0];

print(loadData('./TimeSeriesPredictionTrain.csv'));
