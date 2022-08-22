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
import xgboost

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




param_grid = {'n_estimators': [ 500, 700 ,900, 1000 , 2000 , 3000],
              'colsample_bytree': [0.7, 0.8],
              'max_depth': [1,7,6,5,15,20,25],
              'subsample': [0.6,0.7, 0.8, 0.9] ,
              'learning_rate' :[ 0.01 , 0.05 ,0.1 ,0.5 ,1] 
}


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
    model = xgboost.XGBClassifier()
    # define search space
    search = BayesSearchCV(model, 
                           param_grid, 
                           scoring='accuracy',
                           n_iter=bcvj ,
                           cv=cv_inner ,
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
# Optimal eXtreme Gradient Boost model
trainedmodel = xgboost.XGBClassifier(n_estimators= 800 ,
              colsample_bytree= 0.75 ,
              max_depth= 7,
              subsample=0.7 ,
              learning_rate= 0.5).fit(X_train,y_train)

y_pred_prob = trainedmodel.predict_proba(X_test)[:,1]

# AUC curve

threshold = 0.5
ROC_curve(y_test , y_pred_prob ,
          threshold ,
          roc_out = "../../../ML_results/small_gene_count/classification_performance/xgb_roc.png"  , 
          sensitivity_out ="../../../ML_results/small_gene_count/classification_performance/xgb_sensitivity.txt" )


XGB_Grid_ytest_THR = ((trainedmodel.predict_proba(X_test)[:, 1])>= threshold).astype(int)

# Classification reports 
with open("../../../ML_results/small_gene_count/classification_performance/xgb_report.txt" , 'w') as f:
    for ele in ['Classification report :' ,  classification_report(y_test, XGB_Grid_ytest_THR) ]:
        f.write(str(ele))

# Confusion matrixs  
with open("../../../ML_results/small_gene_count/classification_performance/xgb_confusion_matrix.txt" , 'w') as f:
    for ele in ['Confusion matrix' ,  confusion_matrix(y_test, XGB_Grid_ytest_THR) ]:
        f.write(str(ele))

#Feature importance 
feature_plot(trainedmodel,
             attributeNames , 
             top_features= 50 ,
             figure_out ='../../../ML_results/small_gene_count/feature_importance/xgb_most_important_features.png',
             text_out ='../../../ML_results/small_gene_count/feature_importance/xgb_top_genes.txt' )



# To export to compare_df
xgb_y_test = y_test
xgb_predictions = trainedmodel.predict(X_test)