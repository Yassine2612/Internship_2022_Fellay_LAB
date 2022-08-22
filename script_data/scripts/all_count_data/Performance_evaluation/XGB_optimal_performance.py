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
from skopt import BayesSearchCV

## Importing utility scripts 
import os
import sys
import inspect
src_file_path = os.path.dirname(os.path.abspath("__file__"))
mymodule_dir = os.path.join( src_file_path, '..', '..', 'utility' )
sys.path.append( mymodule_dir )
from evaluateBinaryClassification import *
from evaluate_threshold import *
from feature_plot_rf_xgb import *


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



# Optimal eXtreme Gradient Boost model
trainedmodel = xgboost.XGBClassifier(n_estimators= 700 ,
              colsample_bytree= 0.7864840831382067 ,
              max_depth= 15,
              subsample=0.6 ,
              learning_rate= 0.1).fit(X_train,y_train)

y_pred_prob = trainedmodel.predict_proba(X_test)[:,1]

# AUC curve

threshold = 0.5
ROC_curve(y_test , y_pred_prob ,
          threshold ,
          roc_out = "../../../../ML_results/all_gene_count/classification_performance/xgb_roc.png"  , 
          sensitivity_out ="../../../../ML_results/all_gene_count/classification_performance/xgb_sensitivity.txt" )


XGB_Grid_ytest_THR = ((trainedmodel.predict_proba(X_test)[:, 1])>= threshold).astype(int)

# Classification reports 
with open("../../../../ML_results/all_gene_count/classification_performance/xgb_report.txt" , 'w') as f:
    for ele in ['Classification report :' ,  classification_report(y_test, XGB_Grid_ytest_THR) ]:
        f.write(str(ele))

# Confusion matrixs  
with open("../../../../ML_results/all_gene_count/classification_performance/xgb_confusion_matrix.txt" , 'w') as f:
    for ele in ['Confusion matrix' ,  confusion_matrix(y_test, XGB_Grid_ytest_THR) ]:
        f.write(str(ele))

#Feature importance 
feature_plot(trainedmodel,
             attributeNames , 
             top_features= 50 ,
             figure_out ='../../../../ML_results/all_gene_count/feature_importance/xgb_most_important_features.png',
             text_out ='../../../../ML_results/all_gene_count/feature_importance/xgb_top_genes.txt' )



# To export to compare_df
xgb_y_test = y_test
xgb_predictions = trainedmodel.predict(X_test)