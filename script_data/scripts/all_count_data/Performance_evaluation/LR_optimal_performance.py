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

## Importing utility scripts 
import os
import sys
import inspect
src_file_path = os.path.dirname(os.path.abspath("__file__"))
mymodule_dir = os.path.join( src_file_path, '..', '..', 'utility' )
sys.path.append( mymodule_dir )
from evaluateBinaryClassification import *
from evaluate_threshold import *
from feature_plot_lr_svm import *


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

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=.2, 
                                                    random_state=RS)

## Optimal Logistic regression 
trainedmodel = LogisticRegression(solver='newton-cg' ,
                                  penalty='l2',
                                  C= 10e-3).fit(X_train,y_train)



y_pred_prob = trainedmodel.predict_proba(X_test)[:,1]

# AUC curve

threshold = 0.5
ROC_curve(y_test , y_pred_prob ,
          threshold ,
          roc_out = "../../../../ML_results/all_gene_count/classification_performance/lr_roc.png"  , 
          sensitivity_out ="../../../../ML_results/all_gene_count/classification_performance/lr_sensitivity.txt" )



LR_Grid_ytest_THR = ((trainedmodel.predict_proba(X_test)[:, 1])>= threshold).astype(int)

# Classification reports 
with open("../../../../ML_results/all_gene_count/classification_performance/lr_report.txt" , 'w') as f:
    for ele in ['Classification report :' ,  classification_report(y_test, LR_Grid_ytest_THR) ]:
        f.write(str(ele))

# Confusion matrixs  
with open("../../../../ML_results/all_gene_count/classification_performance/lr_confusion_matrix.txt" , 'w') as f:
    for ele in ['Confusion matrix' ,  confusion_matrix(y_test, LR_Grid_ytest_THR) ]:
        f.write(str(ele))

#Feature importance 
feature_plot(trainedmodel,
             attributeNames , 
             top_features= 50 ,
             figure_out ='../../../../ML_results/all_gene_count/feature_importance/lr_most_important_features.png',
             text_out ='../../../../ML_results/all_gene_count/feature_importance/lr_top_genes.txt' )



# To export to compare_df
lr_y_test = y_test
lr_predictions = trainedmodel.predict(X_test)
