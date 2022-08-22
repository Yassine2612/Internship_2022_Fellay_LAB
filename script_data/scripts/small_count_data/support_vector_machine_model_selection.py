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
from sklearn.svm import SVC


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
from feature_plot_lr_svm import *




X = preprocessing.scale(X) # Scale the imput data 

RS = 20 # RANDOM STATE so that the results are reproducible 



param_grid = {'C': [0.0001,0.001,0.01,0.1,1, 10, 100], 
              'gamma': [10000,1000,100,10,1,0.1,0.01,0.001],
              'kernel': ['rbf', 'poly', 'sigmoid' ,'linear' ]}



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
    model = SVC()
    # define search space
    search = GridSearchCV(model, 
                          param_grid, 
                          scoring='accuracy',
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
# Optimal SVM  model
trainedmodel = trainedSVM = SVC(kernel='linear' ,
                                gamma=1000 ,
                                C =0.1,
                                probability=True).fit(X_train,y_train)

y_pred_prob = trainedmodel.predict_proba(X_test)[:,1]

# AUC curve

threshold = 0.5
ROC_curve(y_test , y_pred_prob ,
          threshold ,
          roc_out = "../../../ML_results/small_gene_count/classification_performance/svm_roc.png"  , 
          sensitivity_out ="../../../ML_results/small_gene_count/classification_performance/svm_sensitivity.txt" )


SVM_Grid_ytest_THR = ((trainedmodel.predict_proba(X_test)[:, 1])>= threshold).astype(int)

# Classification reports 
with open("../../../ML_results/small_gene_count/classification_performance/svm_report.txt" , 'w') as f:
    for ele in ['Classification report :' ,  classification_report(y_test, SVM_Grid_ytest_THR) ]:
        f.write(str(ele))

# Confusion matrixs  
with open("../../../ML_results/small_gene_count/classification_performance/svm_confusion_matrix.txt" , 'w') as f:
    for ele in ['Confusion matrix' ,  confusion_matrix(y_test, SVM_Grid_ytest_THR) ]:
        f.write(str(ele))

#Feature importance 
feature_plot(trainedmodel,
             attributeNames , 
             top_features= 50 ,
             figure_out ='../../../ML_results/small_gene_count/feature_importance/svm_most_important_features.png',
             text_out ='../../../ML_results/small_gene_count/feature_importance/svm_top_genes.txt' )



# To export to compare_df
svm_y_test = y_test
svm_predictions = trainedmodel.predict(X_test)




