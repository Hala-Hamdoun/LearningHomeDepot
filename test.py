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

#takes the datestring and converts it to a number representing the day (1-365)
#value on the right side is untouched ex: "2016-01-12|14" becomes "12|14"
def dateParser(productArray):
    monthOffset = [ 0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365 ];
    parsed_array = [None] * 365;

    for i in range( 0, 365 ):
        if parsed_array[i] == None:
            parsed_array[i] = str(i + 1) + "|0";

    for i in range( 0, len( productArray ) ):
        split_string = productArray[i].split('|');
        numbers_string = split_string[0].split('-');
        parsed_array[ monthOffset[ int( numbers_string[1] ) ] + int( numbers_string[2] ) - 1 ] = str( monthOffset[ int( numbers_string[1] ) ] + int( numbers_string[2] ) ) + '|' + split_string[1];

    return parsed_array

def getFeatureResultFormat(productTriples):
    features = []; #[two days ago views][one day ago views]
    classifications = [] # [third day views];

    for triple in productTriples:
        if(len(triple) == 3):        
            twoAgo = triple[0][1]        
            oneAgo = triple[1][1];        
            features.append([twoAgo, oneAgo]);
            classifications.append(triple[2][1]);

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
Xtr, Ytr = train;
Xte, Yte = test;
        
    
# tf Graph Input
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# Prediction: Get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0.

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # loop over test data
    for i in range(len(Xte)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        # Get nearest neighbor class label and compare it to its true label
        ##print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
        ##    "True Class:", np.argmax(Yte[i]))
        # Calculate accuracy
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print("Done!")
    print("Accuracy:", accuracy)




