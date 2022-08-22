## Utilitary libraries
import matplotlib.pyplot as plt
from itertools import product
from statistics import mean
import numpy as np


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
from Data_processed_for_classification  import *

## Importing utility scripts 
import os
import sys
import inspect
src_file_path = os.path.dirname(os.path.abspath("__file__"))
mymodule_dir = os.path.join( src_file_path, '..', 'utility' )
sys.path.append( mymodule_dir )
from evaluateBinaryClassification import *
from evaluate_threshold import *
from feature_plot_lr_svm import *



X = preprocessing.scale(X) # Scale the imput data 

RS = 20 # RANDOM STATE so that the results are reproducible 

cv_outer = StratifiedKFold(n_splits= 10, shuffle=True)
# enumerate splits
outer_results = list()
for train_ix, test_ix in cv_outer.split(X,y):
    	# split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
        #Baseline model
        # calculate null accuracy (for binary classification problems coded as 0/1)
    print ( 'The Baseline model accuracy is : '  , max(y_test.mean(), 1 - y_test.mean()) )

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
    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results), np.std(outer_results)))



X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=.2, 
                                                    random_state=RS)

## Logistic regression 
trainedmodel = LogisticRegression(solver='newton-cg' ,
                                  penalty='l2',
                                  C= 1000).fit(X_train,y_train)



y_pred_prob = trainedmodel.predict_proba(X_test)[:,1]

# AUC curve

threshold = 0.5
ROC_curve(y_test , y_pred_prob ,
          threshold ,
          roc_out = "../../../ML_results/small_gene_count/classification_performance/lr_roc.png"  , 
          sensitivity_out ="../../../ML_results/small_gene_count/classification_performance/lr_sensitivity.txt" )



LR_Grid_ytest_THR = ((trainedmodel.predict_proba(X_test)[:, 1])>= threshold).astype(int)

# Classification reports 
with open("../../../ML_results/small_gene_count/classification_performance/lr_report.txt" , 'w') as f:
    for ele in ['Classification report :' ,  classification_report(y_test, LR_Grid_ytest_THR) ]:
        f.write(str(ele))

# Confusion matrixs  
with open("../../../ML_results/small_gene_count/classification_performance/lr_confusion_matrix.txt" , 'w') as f:
    for ele in ['Confusion matrix' ,  confusion_matrix(y_test, LR_Grid_ytest_THR) ]:
        f.write(str(ele))

#Feature importance 
feature_plot(trainedmodel,
             attributeNames , 
             top_features= 50 ,
             figure_out ='../../../ML_results/small_gene_count/feature_importance/lr_most_important_features.png',
             text_out ='../../../ML_results/small_gene_count/feature_importance/lr_top_genes.txt' )



# To export to compare_df
lr_y_test = y_test
lr_predictions = trainedmodel.predict(X_test)
