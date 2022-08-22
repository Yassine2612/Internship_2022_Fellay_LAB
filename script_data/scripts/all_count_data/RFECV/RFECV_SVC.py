import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler

# RFECV
from sklearn.feature_selection import RFECV
import yellowbrick

# CV 
from sklearn.model_selection import StratifiedKFold

## Classifiers

from sklearn.svm import SVC
from sklearn import preprocessing

 # Recursive Feature Elimination With Cross-Validation (RFECV)

from RFECV_yellow_brick import rfecv
 
 
import os
import sys
import inspect
src_file_path = os.path.dirname(os.path.abspath("__file__"))
mymodule_dir = os.path.join( src_file_path, '..', 'import_data' )
sys.path.append( mymodule_dir )
from All_counts_data_processed_for_classification import *
 
 

RS =20 # RANDOM STATE so that the results are reproducible

X = preprocessing.scale(X) # Scale the imput data 

# Optimal SVM  model
trainedmodel = SVC(kernel='linear' ,
                                gamma=10000 ,
                                C =0.0001,
                                probability=True)

cv = StratifiedKFold(5)
#Measuring the optimal number of feature to find the optimal accuracy / cost tradeoff of the classifiers 
plt.figure()
visualizer = rfecv(estimator=trainedmodel ,
                   X = X ,
                   y= y ,
                   cv=cv,
                   step=10 ,
                   scoring='accuracy' ,
                   random_state =RS,
                   show=("../../../../ML_results/all_gene_count/feature_importance/SVC_optimal_nbr_features.png"))


visualizer.fit(X, y)        # Fit the data to the visualizer
visualizer.show("../../../../ML_results/all_gene_count/feature_importance/SVC_optimal_nbr_features.png") 
plt.close()

importance = visualizer.estimator_.coef_[0]
sorted_idx = importance.argsort()
optimal_nbr = visualizer.n_features_
plt.figure()
plt.barh(attributeNames[sorted_idx][-optimal_nbr:] ,importance [sorted_idx][-optimal_nbr:] )
plt.xlabel(" Most important features")
plt.ylabel(" Feature importance")
plt.savefig("../../../../ML_results/all_gene_count/feature_importance/most_important_features_SVC.png")
plt.close()
with open("../../../../ML_results/all_gene_count/feature_importance/SVC_most_important_genes.txt" , 'w') as f:
    for gene in  attributeNames[sorted_idx][-optimal_nbr:]:
        f.write(gene)
        f.write('\n')    
        
