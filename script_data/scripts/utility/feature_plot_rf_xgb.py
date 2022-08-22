import matplotlib.pyplot as plt
import numpy as np

#feature plot for RF and XGB

#out : path_to_save_feature_importance_plot

# function that takes a classification model , the feature_names and the numbr of most important features to output 
def feature_plot(classifier, feature_names, top_features= 50 , figure_out ='../../../../ML_results/most_important_features.png'  , text_out = '../../../../ML_results/most_important_genes.txt'):
  sorted_idx = classifier.feature_importances_.argsort()
  plt.figure()
  plt.barh(feature_names[sorted_idx][-top_features:], classifier.feature_importances_[sorted_idx][-top_features:])
  plt.xlabel("Most important features")
  plt.title("Gene importance ")
  feature_names = np.array(feature_names)
  plt.savefig(figure_out)
  plt.close()
  with open( text_out , 'w') as f:
     for ele in  [feature_names[sorted_idx][-top_features:] , feature_names[sorted_idx][-top_features:] ]  :
         f.write(str(ele))
         f.write('\n')
