# -*- coding: utf-8 -*-

import sys

import pandas as pd

df_train = pd.read_csv('./data/origin_train.csv', encoding='utf-8', engine='python')
q1 = df_train['question1'].values
q2 = df_train['question2'].values
label = df_train['label'].values
category = df_train['category'].values
dict_1 = {}
dict_2 = {}
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
    else:
        if dict_2.get(q1[i], -1) == -1:
            dict_2[q1[i]] = [q2[i]]
        else:
            dict_2[q1[i]].append(q2[i])
        if dict_2.get(q2[i], -1) == -1:
            dict_2[q2[i]] = [q1[i]]
        else:
            dict_2[q2[i]].append(q1[i])

    if i % 5000 == 0:
        sys.stdout.flush()
        sys.stdout.write('#')
print(len(dict_1))

df_result = pd.read_csv('./data/result.csv', encoding='utf-8', engine='python', index_col='id')
df_test = pd.read_csv('./data/noextension/test.csv', encoding='utf-8', engine='python')
q1_test = df_test['question1'].values
q2_test = df_test['question2'].values
category_test = df_test['category'].values
id_test = df_test['id']

cnt = 0
for i in range(0, df_test.shape[0]):
    # print(q1_test[i], q2_test[i], id_test[i], category_test[i])
    list1 = dict_1.get(q1_test[i], -1)
    list2 = dict_1.get(q2_test[i], -1)
    ct = category_test[i]
    if list1 != -1:
        if list2 != -1:
            if len(set(list1).intersection(set(list2))) != 0 and dict_ct.get(q1_test[i], -1) == dict_ct.get(q2_test[i], -1) and dict_ct.get(q1_test[i], -1) != -1:
                df_result.iloc[id_test[i]]['label'] = 1
                print(3)

    # 找到每个q1相似问题q在训练集中是否存在q!=q2，从而推出q1!=q2
    if list1 != -1:
        for q in list1:
            neq_list = dict_2.get(q, -1)
            if neq_list != -1:
                if q2_test[i] in neq_list:
                    df_result.iloc[id_test[i]]['label'] = 0
                    print(1)
    # 同理q2
    if list2 != -1:
        for q in list2:
            neq_list = dict_2.get(q, -1)
            if neq_list != -1:
                if q1_test[i] in neq_list:
                    df_result.iloc[id_test[i]]['label'] = 0
                    print(2)

df_result.to_csv('./data/post_result.csv')
