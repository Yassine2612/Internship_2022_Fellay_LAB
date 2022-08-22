import pandas as pd 
import numpy as np


## Importing utility scripts 
import os
import sys
import inspect
src_file_path = os.path.dirname(os.path.abspath("__file__"))
mymodule_dir = os.path.join( src_file_path, '..', 'utility' )
sys.path.append( mymodule_dir )
from evaluateBinaryClassification import *

#Importing classifier predictions
from LR_optimal_performance import lr_predictions , lr_y_test
from RF_optimal_performance import rf_predictions , rf_y_test
from SVM_optimal_performance import svm_predictions , svm_y_test
from XGB_optimal_performance import xgb_predictions , xgb_y_test
from MLP_optimal_performance import mlp_predictions , mlp_y_test

from Ensembl_classifier import evc_predictions , evc_y_test




## Creating the classification performance table
Methods = ['LR','RF','SVM','XGB' ,'MLP','EVC']
Metrics = ['Accuracy','Recall','Precision','Fscore']
compare_df = pd.DataFrame(index = Methods, columns = Metrics)

#Fill performance table
compare_df.loc['LR'] = evaluateBinaryClassification(lr_predictions,lr_y_test)
compare_df.loc['RF'] = evaluateBinaryClassification(rf_predictions,rf_y_test)
compare_df.loc['SVM'] = evaluateBinaryClassification(svm_predictions,svm_y_test)
compare_df.loc['XGB'] = evaluateBinaryClassification(xgb_predictions,xgb_y_test)
compare_df.loc['MLP'] = evaluateBinaryClassification(mlp_predictions,mlp_y_test)
compare_df.loc['EVC'] = evaluateBinaryClassification(evc_predictions,evc_y_test)

# Confusion matrixs  
with open("../../../ML_results/small_gene_count/classification_performance/classification_perf_summary.txt" , 'w') as f:
    for ele in compare_df :
        f.write(str(ele))