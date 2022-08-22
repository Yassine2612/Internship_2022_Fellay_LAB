## Utilitary libraries
import matplotlib.pyplot as plt
from itertools import product
from statistics import mean
import numpy as np
import warnings


## Sklearn classification metrics  libraries
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn import metrics


from sklearn import preprocessing
from sklearn.inspection import permutation_importance


## Classifiers
from sklearn.ensemble import RandomForestClassifier


## Two layer CV
from sklearn.model_selection import KFold , ShuffleSplit , GridSearchCV , StratifiedKFold ,RandomizedSearchCV

from skopt import BayesSearchCV

## Importing data  
import os
import sys
import inspect
src_file_path = os.path.dirname(os.path.abspath("__file__"))
mydata_dir = os.path.join( src_file_path, '..', 'import_data' )
sys.path.append( mydata_dir )
from All_counts_data_processed_for_classification  import *

##Prevent printing warnings
warnings.filterwarnings("ignore")

X = preprocessing.scale(X) # Scale the imput data 

RS = 20 # RANDOM STATE so that the results are reproducible 
i = 0
param_grid = {'n_estimators': [10,30,50,70,100,150,200],# Number of trees in random forest
                   'max_features': ['auto', 'sqrt'], # Number of features to consider at every split
                   'max_depth':[10,20,30,40,50,60,70,80,90,100,110],# Maximum number of levels in tree
                   'min_samples_split': [2, 5, 7, 10],  # Minimum number of samples required to split a node
                   'min_samples_leaf': [1,2,3,4,5],  # Minimum number of samples required at each leaf node
                   'bootstrap':[True, False]}
#No. of jobs
gcvj = np.cumsum([len(x) for x in param_grid.values()])[-1]
bcvj = int(gcvj)



cv_outer =StratifiedKFold(n_splits=10, shuffle=True, random_state=RS )
# enumerate splits
outer_results = list()
for train_ix, test_ix in cv_outer.split(X,y ):
    	# split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
        
    y_train, y_test = y[train_ix], y[test_ix]
        
    ++i   
    if i == 1 :
        with open("../../../../ML_results/all_gene_count/hyper_param_tunning/rf_tunning_process.txt" , 'w') as f:
            for ele in  [" Baseline model accuracy" , str( max(y_test.mean(), 1 - y_test.mean()) ) ]:
                f.write(str(ele))
    else :
        with open("../../../../ML_results/all_gene_count/hyper_param_tunning/rf_tunning_process.txt" , 'a') as f:
            f.write('\n')
            for ele in  [" Baseline model accuracy" , str( max(y_test.mean(), 1 - y_test.mean()) ) ]:
                f.write(str(ele))
                
    	# configure the cross-validation procedure
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    	# define the model
    model = RandomForestClassifier()
    # define search space
    search = BayesSearchCV(model, 
                           param_grid,
                           scoring='accuracy', 
                           cv=cv_inner,
                           n_iter=bcvj ,
                           refit=True)
    	# execute search
    result = search.fit(X_train, y_train)
    	# get the best performing model fit on the whole training set
    best_model = result.best_estimator_
    	# evaluate model on the hold out dataset
    yhat = best_model.predict(X_test)
    	# evaluate the model
    acc = accuracy_score(y_test, yhat)
    	# store the result
    outer_results.append(acc)
    	# report progress
    with open("../../../../ML_results/all_gene_count/hyper_param_tunning/rf_tunning_process.txt" , 'a') as f:
        f.write('\n') 
        for ele in  [result.best_score_, result.best_params_ ]:
            f.write(str(ele))
            
# summarize the estimated performance of the model
with open("../../../../ML_results/all_gene_count/hyper_param_tunning/rf_tunning_process.txt" , 'a') as f:
    f.write('\n')
    for ele in  [mean(outer_results), np.std(outer_results) ]:
        f.write(str(ele))


