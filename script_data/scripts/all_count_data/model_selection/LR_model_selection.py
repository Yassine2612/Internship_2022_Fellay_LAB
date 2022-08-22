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
from sklearn.linear_model import LogisticRegression

## Two layer CV
from sklearn.model_selection import KFold , ShuffleSplit , GridSearchCV , StratifiedKFold ,RandomizedSearchCV

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
i=0
cv_outer = StratifiedKFold(n_splits= 10, shuffle=True)
# enumerate splits
outer_results = list()
for train_ix, test_ix in cv_outer.split(X,y):
    	# split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
        #Baseline model
        # calculate null accuracy (for binary classification problems coded as 0/1)
    ++i   
    if i == 1 :
        with open("../../../../ML_results/all_gene_count/hyper_param_tunning/lr_tunning_process.txt" , 'w') as f:
            for ele in  [" Baseline model accuracy" , str( max(y_test.mean(), 1 - y_test.mean()) ) ]:
                f.write(str(ele))
    else :
        with open("../../../../ML_results/all_gene_count/hyper_param_tunning/lr_tunning_process.txt" , 'a') as f:
            f.write('\n')
            for ele in  [" Baseline model accuracy" , str( max(y_test.mean(), 1 - y_test.mean()) ) ]:
                f.write(str(ele))
                
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True)
    	# define the model
    model = LogisticRegression( penalty='l2' )
    	# define search space
    parameters = { 
        'C'       : np.logspace(-3,3,7) , #strength
        'solver' : ['newton-cg', 'lbfgs', 'liblinear' ]
    }
    	# define search
    search = GridSearchCV(model,
                          parameters,
                          scoring='accuracy',
                          cv=cv_inner, 
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
    with open("../../../../ML_results/all_gene_count/hyper_param_tunning/lr_tunning_process.txt" , 'a') as f:
        f.write('\n') 
        for ele in  [result.best_score_, result.best_params_ ]:
            f.write(str(ele))
            
# summarize the estimated performance of the model
with open("../../../../ML_results/all_gene_count/hyper_param_tunning/lr_tunning_process.txt" , 'a') as f:
    f.write('\n')
    for ele in  [mean(outer_results), np.std(outer_results) ]:
        f.write(str(ele))
       


