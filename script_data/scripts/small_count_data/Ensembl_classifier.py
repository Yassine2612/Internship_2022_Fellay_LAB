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

from sklearn.neural_network import MLPClassifier

## Two layer CV
from Data_processed_for_classification  import *


## Classifiers
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


## Importing utility scripts 
import os
import sys
import inspect
src_file_path = os.path.dirname(os.path.abspath("__file__"))
mymodule_dir = os.path.join( src_file_path, '..', 'utility' )
sys.path.append( mymodule_dir )
from evaluateBinaryClassification import *
from evaluate_threshold import *
from feature_plot_mlp import *


RS = 20
warnings.filterwarnings("ignore") # Prevent printing warnings
X = preprocessing.scale(X) # Scalig the data


## Optimal Logistic regression 
trained_logistic_regression = LogisticRegression(solver='newton-cg' ,
                                  penalty='l2',
                                  C= 1000)

# Optimal RF  model
trained_random_forest = RandomForestClassifier( max_features= 'auto' ,
                                       n_estimators=70 ,
                                       min_samples_split =7 ,
                                       max_depth =70 ,
                                       min_samples_leaf =2 , 
                                       bootstrap=False  )

# Optimal SVM  model
trained_SVC = trainedSVM = SVC(kernel='linear' ,
                                gamma=1000 ,
                                C =0.1,
                                probability=True)
# Optimal XGB  model
trained_xgboost = xgboost.XGBClassifier(n_estimators= 800 ,
              colsample_bytree= 0.75 ,
              max_depth= 7,
              subsample=0.7 ,
              learning_rate= 0.5)

# Optimal MLP  model
trained_MLP = MLPClassifier(hidden_layer_sizes=7,
                              activation='relu' , 
                              learning_rate='constant', 
                              solver = 'lbfgs' )



evc = VotingClassifier( estimators= [  ('lr',trained_logistic_regression) ,('rf',trained_random_forest) ,('svm',trained_SVC) , ('xgb' ,trained_xgboost) ,('mlp' , trained_MLP)  ] ,  
                                     voting = 'soft')
                                     
                                     
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=RS)

trained_EVC = evc.fit(X_train,y_train)

y_pred_prob = trained_EVC.predict_proba(X_test)[:,1]

# AUC curve

threshold = 0.5
ROC_curve(y_test , y_pred_prob ,
          threshold ,
          roc_out = "../../../ML_results/small_gene_count/classification_performance/evc_roc.png"  , 
          sensitivity_out ="../../../ML_results/small_gene_count/classification_performance/evc_sensitivity.txt" )


EVC_Grid_ytest_THR = ((trained_EVC.predict_proba(X_test)[:, 1])>= threshold).astype(int)

# Classification reports 
with open("../../../ML_results/small_gene_count/classification_performance/evc_report.txt" , 'w') as f:
    for ele in ['Classification report :' ,  classification_report(y_test, EVC_Grid_ytest_THR) ]:
        f.write(str(ele))

# Confusion matrixs  
with open("../../../ML_results/small_gene_count/classification_performance/evc_confusion_matrix.txt" , 'w') as f:
    for ele in ['Confusion matrix' ,  confusion_matrix(y_test, EVC_Grid_ytest_THR) ]:
        f.write(str(ele))

#Feature importance 
feature_plot(classifier= trained_EVC,
             feature_names =attributeNames ,
             x_train = X_train ,
             y_train = y_train , 
             top_features= 50 ,
             figure_out ='../../../ML_results/small_gene_count/feature_importance/evc_most_important_features.png',
             text_out ='../../../ML_results/small_gene_count/feature_importance/evc_top_genes.txt' )

# To export to compare_df
evc_y_test = y_test
evc_predictions = trained_EVC.predict(X_test)


