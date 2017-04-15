from __future__ import print_function
from helperFunctions import *

import numpy as np
import tensorflow as tf

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
        
Xtr = np.asarray(Xtr);
Xte = np.asarray(Xte);    
Ytr = np.asarray(Ytr);
Yte = np.asarray(Yte);

# tf Graph Input
xtr = tf.placeholder("float", [None, 2])
xte = tf.placeholder("float", [2])

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
