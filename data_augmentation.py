# -*- coding: utf-8 -*-

import copy
import os
import numpy as np
import random
import sys

import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

""" Naive train test split. """
# df = pd.read_csv('./data/aug_train.csv', encoding='utf-8', engine='python')
# df = shuffle(df)
# train_df = df[2000:]
# dev_df = df[:2000]
# train_df.to_csv('./data/origin_train.csv', index=False)
# dev_df.to_csv('./data/dev.csv', index=False)

""" KFold """
# parent_directory = './data/extension/'
# df = pd.read_csv(parent_directory + 'final_train.csv', encoding='utf-8', engine='python')
# kFold = KFold(n_splits=10, shuffle=True, random_state=12345)
# folds = kFold.split(df)
# for i in range(10):
#     os.makedirs(parent_directory + str(i))
# for i, (train, dev) in enumerate(folds):
#     df.iloc[train].to_csv(parent_directory + str(i) + '/origin_train.csv', index=False)
#     df.iloc[dev].to_csv(parent_directory + str(i) + '/dev.csv', index=False)

""" Data extension by questions similarity. """
df_train = pd.read_csv('./data/origin_train.csv', encoding='utf-8', engine='python')
q1 = df_train['question1'].values
q2 = df_train['question2'].values
label = df_train['label'].values
category = df_train['category'].values
dict_1 = {}
dict_ct = {}
for i in range(0, df_train.shape[0]):
    dict_ct[q1[i]] = category[i]
    dict_ct[q2[i]] = category[i]
    if label[i] == 1:
        if dict_1.get(q1[i], -1) == -1:
            dict_1[q1[i]] = [q2[i]]
        else:
            dict_1[q1[i]].append(q2[i])
        if dict_1.get(q2[i], -1) == -1:
            dict_1[q2[i]] = [q1[i]]
        else:
            dict_1[q2[i]].append(q1[i])
    if i % 5000 == 0:
        sys.stdout.flush()
        sys.stdout.write('#')
print(len(dict_1))

listxy = []
for x in dict_1:
    listx = dict_1[x]
    if len(listx) > 1:
        listy = listx[:]
        random.shuffle(listy)
        for x, y in zip(listx, listy):
            if x != y and y not in dict_1[x] and x not in dict_1[y]:
                if dict_ct[x] != dict_ct[y]:
                    ct = 'wrong'
                    listxy.append([x, y, 0, ct])
                else:
                    ct = dict_ct[x]
                    listxy.append([x, y, 1, ct])
print(len(listxy))
random.shuffle(listxy)
df_ext = pd.DataFrame(listxy)
df_ext.columns = ['question1', 'question2', 'label', 'category']
df_ext.to_csv('./data/extension/ext_train.csv', index=False)

""" Produce negative samples and ombine extension dataset. """
df_ext_train = pd.read_csv('./data/extension/ext_train.csv')
temp_q1 = df_train['question1'].values.copy()
temp_q2 = df_train['question2'].values.copy()
np.random.shuffle(temp_q1)
np.random.shuffle(temp_q2)
temp_df = pd.DataFrame()
temp_df['label'] = np.zeros(temp_q1.shape[0], dtype=int)
temp_df['question1'] = temp_q1
temp_df['question2'] = temp_q2
category_col = []
for i in range(len(temp_df)):
    if dict_ct[temp_df.iloc[i]['question1']] == dict_ct[temp_df.iloc[i]['question2']]:
        category_col.append(dict_ct[temp_df.iloc[i]['question1']])
    else:
        category_col.append('wrong')
temp_df['category'] = category_col
temp_df = temp_df.sample(n=int(df_ext_train.shape[0]*0.8))
df_train = pd.concat([df_train, df_ext_train, temp_df], sort=False)
df_train = df_train.drop_duplicates(['question1', 'question2']).reset_index(drop=True)
df_train.columns = ['question1', 'question2', 'label', 'category']
df_train.to_csv('./data/extension/final_train.csv', index=False)
print('Complete.')

parent_directory = './data/extension/'
df = pd.read_csv(parent_directory + 'final_train.csv', encoding='utf-8', engine='python')
kFold = KFold(n_splits=10, shuffle=True, random_state=12345)
folds = kFold.split(df)
for i in range(10):
    os.makedirs(parent_directory + str(i))
for i, (train, dev) in enumerate(folds):
    df.iloc[train].to_csv(parent_directory + str(i) + '/origin_train.csv', index=False)
    df.iloc[dev].to_csv(parent_directory + str(i) + '/dev.csv', index=False)


category_map = {
    'aids':             '艾滋病',
    'breast_cancer':    '乳腺癌',
    'diabetes':         '糖尿病',
    'hepatitis':        '乙肝',
    'hypertension':     '高血压'
}
