import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from deap import creator, base, tools, algorithms
from sklearn.ensemble import RandomForestClassifier  , GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB , MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score



# Random Forest Classifier
RF_CLR = RandomForestClassifier(n_estimators= 50 ,random_state=42 , n_jobs=-1)
def random_forest_classifier_train(X_train ,  y_test):
    RF_CLR.fit(X_train, y_train)
    
def random_forest_classifier_test( X_test , y_test):
    y_pred = RF_CLR.predict(X_test)
    print('Accuracy on test set: ')
    print(accuracy_score(y_test, y_pred))
    print(f1_score(y_test,y_pred))
    print(precision_score(y_test,y_pred))
    print(recall_score(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))

# K Nearest Neighbors
KNN_CLR = KNeighborsClassifier(n_neighbors= 200,weights ='uniform',algorithm='auto')

def KNN_classifier_train(X_train, y_train ):
    KNN_CLR.fit(X_train, y_train)
    
def KNN_classifier_test(X_test , y_test ):
    y_pred = KNN_CLR.predict(X_test)
    print('Accuracy on test set: ')
    print(accuracy_score(y_test, y_pred))
    print(f1_score(y_test,y_pred))
    print(precision_score(y_test,y_pred))
    print(recall_score(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))

# Gradient Boosting Decision Tree
GBDT_CLR = GradientBoostingClassifier(n_estimators=10,max_depth=2, random_state=42)

def GBDT_classifier_train(X_train, y_train ):
    GBDT_CLR.fit(X_train, y_train)
    
def GBDT_classifier_test(X_test , y_test ):
    y_pred = GBDT_CLR.predict(X_test)
    print('Accuracy on test set: ')
    print(accuracy_score(y_test, y_pred))
    print(f1_score(y_test,y_pred))
    print(precision_score(y_test,y_pred))
    print(recall_score(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))

# xExtreme Gradient Boosting
'''  XGBOOST '''
param = {
    'eval_metric':'mlogloss',
    'eta': 0.4, 
    'max_depth':2,  
    'objective': 'multi:softprob',  
    'num_class': 2 
    }
def Xgboost_classifier_train(X_train, y_train ):
    D_train = xgb.DMatrix(X_train, label=y_train)
    global model
    model = xgb.train(param, D_train)

def Xgboost_classifier_test(X_test , y_test):
    D_test = xgb.DMatrix(X_test, label=y_test)
    preds = model.predict(D_test)
    best_preds = np.asarray([np.argmax(line) for line in preds])
    print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
    print(f1_score(y_test,y_pred))
    print(precision_score(y_test,best_preds))
    print(recall_score(y_test,best_preds))
    print(confusion_matrix(y_test,best_preds))
    print(classification_report(y_test,best_preds))