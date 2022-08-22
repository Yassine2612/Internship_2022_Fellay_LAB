import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


    # AUC curve
    # IMPORTANT: first argument is true values, second argument is predicted probabilities
def ROC_curve(y_test , y_pred_prob ,
              threshold = 0.5 ,
              roc_out = "../../../../ML_results/roc.png" ,
              sensitivity_out = "../../../../ML_results/sensitivity.txt") : 
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlim([-0.05, 1.1])
    plt.ylim([-0.05, 1.1])
    plt.title('ROC curve for severe COVID19 classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.savefig(roc_out)
    plt.close()
    #  function that accepts a threshold and prints sensitivity and specificity
    with open( sensitivity_out , 'w') as f:
        for ele in  [  'Sensitivity:' , str (tpr[thresholds > threshold][-1]) , '\n' , 'Specificity:',  str(1 - fpr[thresholds > threshold][-1] ) ]   :
            f.write(str(ele))
    
