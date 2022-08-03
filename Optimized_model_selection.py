## Utilitary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix
from itertools import product
from statistics import mean
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.inspection import permutation_importance
from sklearn import metrics

## Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


## Two layer CV
from sklearn.model_selection import KFold , ShuffleSplit , GridSearchCV , StratifiedKFold ,RandomizedSearchCV
from Data_processed_for_classification  import *


def feature_plot(classifier, feature_names, top_features= 15):
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
 plt.show()
 print("List of the TOP genes :" ,feature_names[top_coefficients])


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

Methods = ['LR','RF','SVM','XGB' ,'MLP']
Metrics = ['Accuracy','Recall','Precision','Fscore']
compare_df = pd.DataFrame(index = Methods, columns = Metrics)

X = preprocessing.scale(X) # Scale the imput data 
RS = 20 # RANDOM STATE so that the results are reproducible 

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
    model = LogisticRegression( penalty='l2' )
    	# define search space
    parameters = { 
        'C'       : np.logspace(-3,3,7) ,
        'solver' : ['newton-cg', 'lbfgs', 'liblinear' ]
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



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=RS)

## Logistic regression 
trainedmodel = LogisticRegression(solver='newton-cg' ,penalty='l2', C= 1000).fit(X_train,y_train)
predictions = trainedmodel.predict(X_test)

y_pred_prob = trainedmodel.predict_proba(X_test)[:,1]
# AUC curve
# IMPORTANT: first argument is true values, second argument is predicted probabilities
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([-0.05, 1.1])
plt.ylim([-0.05, 1.1])
plt.title('ROC curve for severe COVID19 classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

thresholds_ =[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.95]
for i in thresholds_ :
    print(evaluate_threshold(i))


threshold = 0.8
LR_Grid_ytest_THR = ((trainedmodel.predict_proba(X_test)[:, 1])>= threshold).astype(int)

# Classification reports 
print('Classification report for the regular threshold of 0.5 :')
print(classification_report(y_test, trainedmodel.predict(X_test)))
print("-------------------------------------------------------------------")
print('Classification report for the new threshold of 0.1 :')
print(classification_report(y_test, LR_Grid_ytest_THR))


# Confusion matrixs  
print('Confusion matrix for the regular threshold of 0.8 :')
print(confusion_matrix(y_test, trainedmodel.predict(X_test)))
print("-------------------------------------------------------------------")
print('Confusion matrix for the new threshold of 0.8 :')
print(confusion_matrix(y_test, LR_Grid_ytest_THR))


#Feature importance 
plt.figure()
feature_plot(trainedmodel, attributeNames , top_features= 15)

#Fill performance table
evaluateBinaryClassification(predictions,y_test)
compare_df.loc['LR'] = evaluateBinaryClassification(predictions,y_test)




## Random Forest  

cv_outer =StratifiedKFold(n_splits=10, shuffle=True, random_state=RS )
# enumerate splits
outer_results = list()
for train_ix, test_ix in cv_outer.split(X,y ):
    	# split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
        
    y_train, y_test = y[train_ix], y[test_ix]
        
    print ( 'The Baseline model accuracy is : '  , max(y_test.mean(), 1 - y_test.mean()) )
    	# configure the cross-validation procedure
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    	# define the model
    model = RandomForestClassifier()
    	# define search space
    	#space = dict()
        # Number of trees in random forest
    n_estimators = [10,50,100,150,200]
        # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
        # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
    bootstrap = [True, False]
    	#space['n_estimators'] = [50, 100, 150 ,200]
    	#space['max_features'] = [2, 4, 6, 8, 10]
        
    random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
    	# define search
    search = RandomizedSearchCV(model, random_grid, scoring='accuracy', cv=cv_inner, n_iter=20 , refit=True)
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



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=RS)


trainedforest = RandomForestClassifier( max_features= 'auto' , n_estimators=200 ,min_samples_split =5 ,max_depth =70 ,min_samples_leaf =4 , bootstrap=False  ).fit(X_train,y_train)
predictionforest = trainedforest.predict(X_test)

y_pred_prob = trainedforest.predict_proba(X_test)[:,1]
# AUC curve
# IMPORTANT: first argument is true values, second argument is predicted probabilities
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([-0.05, 1.1])
plt.ylim([-0.05, 1.1])
plt.title('ROC curve for severe COVID19 classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

thresholds_ =[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in thresholds_ :
    print("WITH A THRESHOLD OF " , i )
    print(evaluate_threshold(i))


print(confusion_matrix(y_test,predictionforest))
print(classification_report(y_test,predictionforest))

evaluateBinaryClassification(predictionforest,y_test)
compare_df.loc['RF'] = evaluateBinaryClassification(predictionforest,y_test)



sorted_idx = trainedforest.feature_importances_.argsort()
plt.figure()
plt.barh(attributeNames[sorted_idx][-30:], trainedforest.feature_importances_[sorted_idx][-30:])
plt.xlabel("Random Forest TOP  important features")
print("List of the TOP genes for RandomForest" , attributeNames[sorted_idx][-30:])



##SVM

param_grid = {'C': [0.0001,0.001,0.01,0.1,1, 10, 100], 
              'gamma': [10000,1000,100,10,1,0.1,0.01,0.001],
              'kernel': ['rbf', 'poly', 'sigmoid' ,'linear' ]}

## SVM 
# The optimal SVM is non linear we therefore can't extract the feature importance 
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=RS)
# enumerate splits
outer_results = list()
for train_ix, test_ix in cv_outer.split(X,y):
    
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    
    print ( 'The Baseline model accuracy is : '  , max(y_test.mean(), 1 - y_test.mean()) )

    	# configure the cross-validation procedure
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    	# define the model
    SVM_model = SVC()
    	# define search
    search = RandomizedSearchCV(SVM_model, param_grid , scoring='accuracy', n_iter=50 , cv=cv_inner, refit=True)
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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=RS)
trainedSVM = SVC(kernel='linear' ,gamma=1000 ,C =0.1,probability=True).fit(X_train,y_train)
y_pred_prob = trainedSVM.predict_proba(X_test)[:,1]
predictions = trainedSVM.predict(X_test)

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
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

thresholds_ =[0.01 ,0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ]
for i in thresholds_ :
    print("WITH A THRESHOLD OF " , i )
    print(evaluate_threshold(i))

threshold=0.3
SVM_Grid_ytest_THR = ((trainedSVM.predict_proba(X_test)[:, 1])>= threshold).astype(int)

# Classification reports 
print('Classification report for the regular threshold of 0.5 :')
print(classification_report(y_test, trainedSVM.predict(X_test)))
print("-------------------------------------------------------------------")
print('Classification report for the new threshold of 0.1 :')
print(classification_report(y_test, SVM_Grid_ytest_THR))


# Confusion matrixs  
print('Confusion matrix for the regular threshold of 0.3 :')
print(confusion_matrix(y_test, trainedSVM.predict(X_test)))
print("-------------------------------------------------------------------")
print('Confusion matrix for the new threshold of 0.3 :')
print(confusion_matrix(y_test, SVM_Grid_ytest_THR))

plt.figure()
feature_plot(trainedSVM, attributeNames)

evaluateBinaryClassification(predictions,y_test)
compare_df.loc['SVM'] = evaluateBinaryClassification(predictions,y_test)



## XGB
params = {     "learning_rate": [0.02, 0.05, 0.1],
               "max_depth": [2, 3, 5],
               "n_estimators":[1000, 2000, 3000] }

cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=RS)
# enumerate splits
outer_results = list()
for train_ix, test_ix in cv_outer.split(X,y):
    	# split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
        
    print ( 'The Baseline model accuracy is : '  , max(y_test.mean(), 1 - y_test.mean()) )
    	# configure the cross-validation procedure
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    	# define the model
    XGBoost_model = xgboost.XGBClassifier()
    	# define search
    search = RandomizedSearchCV(XGBoost_model, params , scoring='accuracy',cv=cv_inner, refit=True)
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



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=RS)

trainedxgboost = xgboost.XGBClassifier(learning_rate = 0.01,
                                     max_depth=2,
                                     n_estimators=1000).fit(X_train,y_train)


predictions = trainedxgboost.predict(X_test)
y_pred_prob = trainedxgboost.predict_proba(X_test)[:,1]

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



print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

thresholds_ =[0.01 ,0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ]
for i in thresholds_ :
    print("WITH A THRESHOLD OF " , i )
    print(evaluate_threshold(i))

threshold=0.4
XGB_Grid_ytest_THR = ((trainedxgboost.predict_proba(X_test)[:, 1])>= threshold).astype(int)

# Classification reports 
print('Classification report for the regular threshold of 0.5 :')
print(classification_report(y_test, trainedxgboost.predict(X_test)))
print("-------------------------------------------------------------------")
print('Classification report for the new threshold of 0.4 :')
print(classification_report(y_test, XGB_Grid_ytest_THR))


# Confusion matrixs  
print('Confusion matrix for the regular threshold of 0.4 :')
print(confusion_matrix(y_test, trainedxgboost.predict(X_test)))
print("-------------------------------------------------------------------")
print('Confusion matrix for the new threshold of 0.4 :')
print(confusion_matrix(y_test, XGB_Grid_ytest_THR))



evaluateBinaryClassification(predictions,y_test)
compare_df.loc['XGB'] = evaluateBinaryClassification(predictions,y_test)

sorted_idx = trainedxgboost.feature_importances_.argsort()
plt.figure()
plt.barh(attributeNames[sorted_idx][-30:] ,trainedxgboost.feature_importances_[sorted_idx][-30:] )
plt.xlabel("XGBoost TOP  important features")
print("List of the TOP genes for XGBoost" , attributeNames[sorted_idx][-30:])



## MLP

# define search space

params = {  "hidden_layer_sizes" :[ 1,5,10 , (5,5) ,(10,10) ] }

cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=RS)

# enumerate splits
outer_results = list()
for train_ix, test_ix in cv_outer.split(X,y):
    	# split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    print ( 'The Baseline model accuracy is : '  , max(y_test.mean(), 1 - y_test.mean()) )
    
    	# configure the cross-validation procedure
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    	# define the model
    model_MLP = MLPClassifier(activation='logistic' , learning_rate='adaptive', solver = 'lbfgs' )
    	# define search
    search = GridSearchCV(model_MLP, params , scoring='accuracy' ,cv=cv_inner ,refit=True)
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



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=RS)

trained_MLP = MLPClassifier(activation='logistic' ,hidden_layer_sizes=(5,5),learning_rate='adaptive' , solver ='lbfgs').fit(X_train,y_train)

y_predict_MLP = trained_MLP.predict(X_test) 
y_pred_prob = trained_MLP.predict_proba(X_test)[:,1]

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


threshold=0.4
MLP_Grid_ytest_THR = ((trained_MLP.predict_proba(X_test)[:, 1])>= threshold).astype(int)

# Classification reports 
print('Classification report for the regular threshold of 0.5 :')
print(classification_report(y_test, trained_MLP.predict(X_test)))
print("-------------------------------------------------------------------")
print('Classification report for the new threshold of 0.4 :')
print(classification_report(y_test, MLP_Grid_ytest_THR))


# Confusion matrixs  
print('Confusion matrix for the regular threshold of 0.4 :')
print(confusion_matrix(y_test, trained_MLP.predict(X_test)))
print("-------------------------------------------------------------------")
print('Confusion matrix for the new threshold of 0.4 :')
print(confusion_matrix(y_test, MLP_Grid_ytest_THR))



results = permutation_importance(trained_MLP, X_train, y_train, scoring= 'accuracy')
# get importance
importance = results.importances_mean
sorted_idx = importance.argsort()
print (importance)
plt.barh(attributeNames[sorted_idx][-50:] ,importance [sorted_idx][-50:] )
plt.xlabel("MLP TOP  important features")
print("List of the TOP genes for MLP" , attributeNames[sorted_idx][-50:])
print(confusion_matrix(y_test,y_predict_MLP))
print(classification_report(y_test,y_predict_MLP))

evaluateBinaryClassification(y_predict_MLP,y_test)
compare_df.loc['MLP'] = evaluateBinaryClassification(y_predict_MLP,y_test)

print(compare_df)
