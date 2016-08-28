from __future__ import print_function
import pandas as pd
#import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from datetime import datetime


def log_reg(X, y, grid):
    """ Logistic Regression """
    
    print('Training logistic classifier ...')
    lr = LogisticRegression()
    clf = GridSearchCV(lr, param_grid={'C': grid}, cv=3, verbose=2)
    clf.fit(X, y['outcome'])
    
    return clf
    
    
def rand_forest(X, y, num_trees=10, max_feat='auto'):
    """ Random Forest """
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y['outcome'], test_size=0.20, 
                                                    random_state=56)
    print('Random Forest Classifier ...')
    rf = RandomForestClassifier(n_estimators=num_trees, max_features=max_feat, verbose=2)
    rf.fit(Xtrain, ytrain)
    yprob = rf.predict_proba(Xtest)
    print('ROC AUC score: ', roc_auc_score(ytest, yprob[:,1]))
    
    return rf
    

if __name__ == '__main__':
    
    data_dir = '../processed_data/'
    
    print('Loading train data ...')
    X = pd.read_csv(data_dir+'X.csv')
    y = pd.read_csv(data_dir+'y.csv')
    
#    clf = log_reg(X, y, np.logspace(-1,1,3))
    clf = rand_forest(X, y, 100)
    
    print('Loading test data ...')
    Xsub = pd.read_csv(data_dir+'Xsub.csv')
    ysub = clf.predict_proba(Xsub)
    
    Sub = pd.read_csv(data_dir+'act_id.csv')
    Sub['outcome'] = ysub[:,1]
    
    print('\nSave submission...')
    savefile = '../submissions/submit-' + str(datetime.now()).split()[0] + '.csv'
    Sub.to_csv(savefile,index=False)
    
    