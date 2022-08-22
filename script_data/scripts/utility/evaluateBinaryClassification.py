from sklearn import metrics
import pandas as pd
import numpy as np
#  function that takes the model prediction and the actual response variable to compute all desired performance metrics 
# Acuracy, Recall, Precision, FScore

def evaluateBinaryClassification(predictions, actuals):
    contigency = pd.crosstab(actuals,predictions)
    print(contigency)
    TP = contigency[0][0]
    TN = contigency[1][1] 
    FP = contigency[0][1]
    FN = contigency[1][0]
    n = contigency.sum().sum()

    Acuracy = (TP + TN)/n
    Recall = TP/(TP+FN)
    Precision = TP/(TP+FP)
    FScore = 2*Recall*Precision/(Recall+Precision)
    
    return Acuracy, Recall, Precision, FScore