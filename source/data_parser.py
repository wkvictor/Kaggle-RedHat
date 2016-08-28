#!usr/bin/python

""" Incorrect processing of categorical data """

from __future__ import print_function
import pandas as pd
#import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocess_people(filename):
    
    print('Preprocess ' + filename)    
    people = pd.read_csv(filename, encoding='utf-8')
    
    people['group'] = people['group_1'].apply(lambda x: int(x.split()[1]))
    people.drop(['date','group_1'], inplace=True, axis=1)
    
    str_col = ['char_' + str(i) for i in range(1,10)]
    people[str_col] = people[str_col].applymap(lambda x: int(x.split()[1]))
        
    bool_col = ['char_' + str(i) for i in range(10,38)]
    people[bool_col] = people[bool_col].applymap(lambda x: int(x))
        
    all_col = ['group'] + str_col + bool_col + ['char_38']
    people[all_col] = MinMaxScaler().fit_transform(people[all_col])
    
    return people
    
    
def preprocess_activity(filename):
    
    print('Preprocess ' + filename)
    
    act = pd.read_csv(filename, encoding='utf-8')
    act.drop(['date'], inplace=True, axis=1)
    act = act.fillna('type 0')
    
    cols = ['char_' + str(i) for i in range(1,11)] + ['activity_category']
    act[cols] = act[cols].applymap(lambda x: int(x.split()[1]))
    act[cols] = MinMaxScaler().fit_transform(act[cols])
    
    return act
    
    
def data_merge(data_dir, save_dir):
    
    people = preprocess_people(data_dir+'people.csv')
    act_train = preprocess_activity(data_dir+'act_train.csv')
    act_test = preprocess_activity(data_dir+'act_test.csv')
    
    print('Start merging data...')
    X = pd.merge(act_train, people, on='people_id', how='left')
    y = X[['outcome']]
    X.drop(['people_id','activity_id','outcome'], inplace=True, axis=1)
    print('X shape: ', X.shape, 'y shape: ', y.shape)
    
    test_id = act_test[['activity_id']]
    Xsub = pd.merge(act_test, people, on='people_id', how='left')
    Xsub.drop(['people_id','activity_id'], inplace=True, axis=1)
    print('Xsub shape: ', Xsub.shape, 'test_id shape: ', test_id.shape)
        
    print('Saving data to csv...')
    X.to_csv(save_dir+'X.csv', index=False)
    y.to_csv(save_dir+'y.csv', index=False)
    
    Xsub.to_csv(save_dir+'Xsub.csv', index=False)
    test_id.to_csv(save_dir+'act_id.csv', index=False)


if __name__ == '__main__':
    
    data_dir = '../input/'
    save_dir = '../processed_data/'
    data_merge(data_dir, save_dir)
    

    