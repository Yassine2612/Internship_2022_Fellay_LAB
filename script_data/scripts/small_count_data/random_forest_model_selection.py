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
from Data_processed_for_classification  import *
from skopt import BayesSearchCV


##Prevent printing warnings
warnings.filterwarnings("ignore")

## Importing utility scripts 
import os
import sys
import inspect
src_file_path = os.path.dirname(os.path.abspath("__file__"))
mymodule_dir = os.path.join( src_file_path, '..', 'utility' )
sys.path.append( mymodule_dir )
from evaluateBinaryClassification import *
from evaluate_threshold import *
from feature_plot_rf_xgb import *


X = preprocessing.scale(X) # Scale the imput data 

RS = 20 # RANDOM STATE so that the results are reproducible 

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
        
    print ( 'The Baseline model accuracy is : '  , max(y_test.mean(), 1 - y_test.mean()) )
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
    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
    # summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results), np.std(outer_results)))



X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=.2, 
                                                    random_state=RS)
# Optimal Random Forest model
trainedmodel = RandomForestClassifier( max_features= 'auto' ,
                                       n_estimators=70 ,
                                       min_samples_split =7 ,
                                       max_depth =70 ,
                                       min_samples_leaf =2 , 
                                       bootstrap=False  ).fit(X_train,y_train)

y_pred_prob = trainedmodel.predict_proba(X_test)[:,1]

# AUC curve

threshold = 0.5
ROC_curve(y_test , y_pred_prob ,
          threshold ,
          roc_out = "../../../ML_results/small_gene_count/classification_performance/rf_roc.png"  , 
          sensitivity_out ="../../../ML_results/small_gene_count/classification_performance/rf_sensitivity.txt" )


RF_Grid_ytest_THR = ((trainedmodel.predict_proba(X_test)[:, 1])>= threshold).astype(int)

# Classification reports 
with open("../../../ML_results/small_gene_count/classification_performance/rf_report.txt" , 'w') as f:
    for ele in ['Classification report :' ,  classification_report(y_test, RF_Grid_ytest_THR) ]:
        f.write(str(ele))

# Confusion matrixs  
with open("../../../ML_results/small_gene_count/classification_performance/rf_confusion_matrix.txt" , 'w') as f:
    for ele in ['Confusion matrix' ,  confusion_matrix(y_test, RF_Grid_ytest_THR) ]:
        f.write(str(ele))

#Feature importance 
feature_plot(trainedmodel,
             attributeNames , 
             top_features= 50 ,
             figure_out ='../../../ML_results/small_gene_count/feature_importance/rf_most_important_features.png',
             text_out ='../../../ML_results/small_gene_count/feature_importance/rf_top_genes.txt' )



# To export to compare_df
rf_y_test = y_test
rf_predictions = trainedmodel.predict(X_test)

