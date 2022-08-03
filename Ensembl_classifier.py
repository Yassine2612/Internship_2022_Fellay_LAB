import numpy as np
import matplotlib.pyplot as plt
from statistics import mean


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_auc_score, classification_report,confusion_matrix
from sklearn import metrics 
from sklearn.inspection import permutation_importance
from sklearn import preprocessing
from sklearn.metrics import roc_curve
import warnings


## Classifiers
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


from Data_processed_for_classification import *
#from Optimized_model_selection import evaluate_threshold , evaluateBinaryClassification ,feature_plot


# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])


def evaluateBinaryClassification(predictions, actuals):
    contigency = pd.crosstab(actuals,predictions)
    print(contigency)
    TP = contigency[0][0]
    TN = contigency[1][1] 
    FP = contigency[0][1]
    FN = contigency[1][0]
    n = contigency.sum().sum()

    Acuracy = (TP + TN)/n
    Recall = TP/(TP+FN)
    Precision = TP/(TP+FP)
    FScore = 2*Recall*Precision/(Recall+Precision)
    
    return Acuracy, Recall, Precision, FScore


warnings.filterwarnings("ignore")
X = preprocessing.scale(X)


trained_logistic_regression= LogisticRegression(solver='newton' ,
                                                penalty='l2',
                                                C=10e3)

trained_random_forest = RandomForestClassifier( max_features= 'auto' ,
                                               n_estimators=200 ,
                                               min_samples_split =5 ,
                                               max_depth =70 ,
                                               min_samples_leaf =4 , 
                                               bootstrap=False)

trained_SVC = SVC(kernel='linear' ,
                  gamma=10e3 ,
                  C =0.1,
                  probability=True)

trained_xgboost = xgboost.XGBClassifier(learning_rate = 0.01,
                                        max_depth=2,
                                        n_estimators=1000)

trained_MLP = MLPClassifier(activation='logistic' ,
                            hidden_layer_sizes=(5,5),
                            learning_rate='adaptive' , 
                            solver ='lbfgs' )




cv_outer = StratifiedKFold(n_splits= 10, shuffle=True)
# enumerate splits
outer_results = list()
for train_ix, test_ix in cv_outer.split(X,y):
    	# split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
        #Baseline model
        # calculate null accuracy (for binary classification problems coded as 0/1)
    print ( 'The Baseline model accuracy is : '  , max(y_test.mean(), 1 - y_test.mean()) )

    cv_inner = StratifiedKFold(n_splits=5, shuffle=True)
    	# define the model
    model = VotingClassifier(estimators = ('lr' , trained_logistic_regression) , voting = 'soft')
    	# define search space
    parameters = { 
        'estimators': [ (('lr',trained_logistic_regression) ,('rf',trained_random_forest) ,('svm',trained_SVC) , ('xgb' ,trained_xgboost) ,('mlp' , trained_MLP)) ,  
                       (('xgb',trained_xgboost) ,('mlp',trained_MLP)) ]
                }
    	# define search
    search = GridSearchCV(model, parameters, scoring='accuracy', cv=cv_inner, refit=True)
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





evc = VotingClassifier( estimators= [ ('xgb' ,trained_xgboost) , ('mlp' , trained_MLP) ] ,
                       voting = 'soft')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=10)

trained_EVC = evc.fit(X_train,y_train)

y_predict_evc = trained_EVC.predict(X_test) 
y_pred_prob = trained_EVC.predict_proba(X_test)[:,1]

# AUC curves
# IMPORTANT: first argument is true values, second argument is predicted probabilities
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([-0.01, 1.1])
plt.ylim([-0.01, 1.1])
plt.title('ROC curve for severe COVID19 classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


thresholds_ =[0.001 , 0.02 ,0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.95 ]
for i in thresholds_ :
    print("WITH A THRESHOLD OF " , i )
    print(evaluate_threshold(i))




# Classification reports 
print('Classification report for the regular threshold of 0.5 :')
print(classification_report(y_test, trained_EVC.predict(X_test)))
print("-------------------------------------------------------------------")

# Confusion matrixs  
print('Confusion matrix for the regular threshold of 0.5 :')
print(confusion_matrix(y_test, trained_EVC.predict(X_test)))


results = permutation_importance(trained_EVC, X_train, y_train, scoring= 'accuracy')
# get importance
importance = results.importances_mean
sorted_idx = importance.argsort()
plt.figure()
plt.barh(attributeNames[sorted_idx][-30:] ,importance [sorted_idx][-30:] )
plt.xlabel("EVC TOP  important features")
print("List of the TOP genes for EVC" , attributeNames[sorted_idx][-30:])




