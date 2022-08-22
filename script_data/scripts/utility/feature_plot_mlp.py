import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance

#feature plot for MLP

#out : path_to_save_feature_importance_plot

# function that takes a classification model , the feature_names and the numbr of most important features to output 
def feature_plot(classifier, x_train , y_train , feature_names, top_features= 50 , figure_out ='../../../../ML_results/most_important_features.png'  , text_out = '../../../../ML_results/most_important_genes.txt'):
  results = permutation_importance(classifier , x_train, y_train, scoring= 'accuracy')
  # get importance
  importance = results.importances_mean
  sorted_idx = importance.argsort()
  plt.figure()
  plt.barh(feature_names[sorted_idx][-top_features:] ,importance [sorted_idx][-top_features:] )
  plt.xlabel("Most important features")
  plt.title("Gene importance ")
  plt.savefig(figure_out)
  plt.close()
  feature_names = np.array(feature_names)
  with open( text_out , 'w') as f:
     for ele in  feature_names[sorted_idx][-top_features:]   :
         f.write(str(ele))
         f.write('\n')


