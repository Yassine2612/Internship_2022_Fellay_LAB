import matplotlib.pyplot as plt
import numpy as np


#out : path_to_save_feature_importance_plot

# function that takes a classification model , the feature_names and the numbr of most important features to output 
def feature_plot(classifier, feature_names, top_features= 50 , figure_out ='../../../../ML_results/most_important_features.png'  , text_out = '../../../../ML_results/most_important_genes.txt'):
 coef = classifier.coef_.ravel()
 top_positive_coefficients = np.argsort(coef)[-top_features:]
 top_negative_coefficients = np.argsort(coef)[:top_features]
 top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
 plt.figure()
 colors = ['green' if c < 0 else 'blue' for c in coef[top_coefficients]]
 plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
 feature_names = np.array(feature_names)
 plt.xticks(np.arange( 2 * top_features), feature_names[top_coefficients], rotation=90, ha='right')
 plt.tick_params(axis='x', which='major', labelsize=8)
 plt.title("Gene importance ")
 plt.savefig(figure_out)
 plt.close()
 with open( text_out , 'w') as f:
     for ele in  feature_names[top_coefficients]   :
         f.write(str(ele))
         f.write('\n')