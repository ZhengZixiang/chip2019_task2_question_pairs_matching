# -*- coding: utf-8 -*-

import jieba
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def get_len_diff(data):
    """
    Get the difference of length and normalize by the longest one of question pairs.
    """
    q1_len = data.question1.apply(lambda x: len(x.split(' '))).values
    q2_len = data.question2.apply(lambda x: len(x.split(' '))).values
    len_diff = np.abs(q1_len - q2_len) / np.max([q1_len, q2_len], axis=0)
    return len_diff


def get_num_common_units(data):
    """
    Get the common units(words or chars) in q1 and q2.
    """
    q1_unit_set = data.question1.apply(lambda x: x.split(' ')).apply(set).values
    q2_unit_set = data.question2.apply(lambda x: x.split(' ')).apply(set).values
    result = [len(q1_unit_set[i] & q2_unit_set[i]) for i in range(len(q1_unit_set))]
    result = pd.DataFrame(result, index=data.index)
    result.columns = ['num_common_units']
    return result


def get_common_units_ratio(data):
    q1_unit_set = data.question1.apply(lambda x: x.split(' ')).apply(set).values()
    q2_unit_set = data.question2.apply(lambda x: x.split(' ')).apply(set).values()
    q1_len = data.question1.apply(lambda x: len(x.split(' '))).values
    q2_len = data.question2.apply(lambda x: len(x.split(' '))).values
    result = [len(q1_unit_set[i] & q2_unit_set[i])/max(q1_len[i], q2_len[i]) for i in range(len(q1_unit_set))]
    result = pd.DataFrame(result, index=data.index)
    result.columns = ['common_units_ratio']
    return result


def get_tfidf_vector(data, vectorizer):
    q1_tfidf = vectorizer.transform(data.question1.values)
    q2_tfidf = vectorizer.transform(data.question2.values)
    return vectorizer.vocabulary_, q1_tfidf, q2_tfidf


def adjust_common_units_ratio_by_tfidf(data, unit2index, q1_tfidf, q2_tfidf):
    adjusted_common_units_ratio = []
    for i in range(q1_tfidf.shape[0]):
        q1_units = {}
        q2_units = {}
        for unit in data.loc[i, 'question1'].lower().split():
            q1_units[unit] = q1_units.get(unit, 0) + 1
        for unit in data.loc[i, 'question2'].lower().split():
            q2_units[unit] = q2_units.get(unit, 0) + 1

        sum_shared_unit_in_q1 = sum([q1_units[u] * q1_tfidf[i, unit2index[u]] for u in q1_units if u in q2_units])
        sum_shared_unit_in_q2 = sum([q2_units[u] * q2_tfidf[i, unit2index[u]] for u in q2_units if u in q1_units])
        sum_total = sum([q1_units[u] * q1_tfidf[i, unit2index[u]] for u in q1_units]) +\
                    sum([q2_units[u] * q2_tfidf[i, unit2index[u]] for u in q2_units])
        if 1e-6 > sum_total:
            adjusted_common_units_ratio.append(0.)
        else:
            adjusted_common_units_ratio.append(1.0 * (sum_shared_unit_in_q1 + sum_shared_unit_in_q2) / sum_total)
    return adjusted_common_units_ratio


def generate_powerful_unit(data):
    """
    Calculate the influence of unit.
    0. the num of unit appeared in question pairs
    1. the ratio of unit appeared in question pairs
    2. the ratio of unit appeared in question pairs labeled 1
    3. the ratio of unit appeared in only one question
    4. the ratio of unit appeared in only one question and pair labeled 1
    5. the ratio of unit appeared in both two questions
    6. the ratio of unit appeared in both two questions and pair labeled 1
    """
    units_power = {}
    for i in data.index:
        label = int(data.loc[i, 'label'])
        q1_units = list(data.loc[i, 'question1'].lower().split())
        q2_units = list(data.loc[i, 'question2'].lower().split())
        all_units = set(q1_units + q2_units)
        q1_units = set(q1_units)
        q2_units = set(q2_units)

        for unit in all_units:
            if unit not in units_power:
                units_power[unit] = [0. for i in range(7)]
            units_power[unit][0] += 1.
            units_power[unit][1] += 1.

            if (unit in q1_units and unit not in q2_units) or (unit not in q1_units and unit in q2_units):
                units_power[unit][3] += 1.
                if 1 == label:
                    units_power[unit][2] += 1.
                    units_power[unit][4] += 1.

            if unit in q1_units and unit in q2_units:
                units_power[unit][5] += 1.
                if 1 == label:
                    units_power[unit][2] += 1.
                    units_power[unit][6] += 1.

    for unit in units_power:
        # calculate ratios
        units_power[unit][1] /= data.shape[0]
        units_power[unit][2] /= data.shape[0]
        if units_power[unit][3] > 1e-6:
            units_power[unit][4] /= units_power[unit][3]
        units_power[unit][4] /= units_power[unit][0]
        if units_power[unit][5] > 1e-6:
            units_power[unit][6] /= units_power[unit][5]
        units_power[unit][5] /= units_power[unit][0]

    sorted_units_power = sorted(units_power.items(), key=lambda d: d[1][0], reverse=True)
    return sorted_units_power


def powerful_units_dside_tag(punit, data, threshold_num, threshold_rate):
    """
    If a powerful units appeared in both questions, the tag was set as 1, otherwise 0.
    """
    punit_dside = []
    punit = filter(lambda x: x[1][0] * x[1][5] >= threshold_num, punit)
    punit_sort = sorted(punit, key=lambda d: d[1][6], reverse=True)
    punit_dside.extend(map(lambda x: x[0], filter(lambda x: x[1][6] >= threshold_rate, punit_sort)))

    punit_dside_tags = []
    for i in data.index:
        tags = []
        q1_units = set(data.loc[i, 'question1'].lower().split())
        q2_units = set(data.loc[i, 'question2'].lower().split())
        for unit in punit_dside:
            if unit in q1_units and unit in q2_units:
                tags.append(1.0)
            else:
                tags.append(0.0)
        punit_dside_tags.append(tags)
    return punit_dside, punit_dside_tags


def powerful_units_oside_tag(punit, data, threshold_num, threshold_rate):
    punit_oside = []
    punit = filter(lambda x: x[1][0] * x[1][3] >= threshold_num, punit)
    punit_oside.extend(map(lambda x: x[0], filter(lambda x: x[1][4] >= threshold_rate, punit)))

    punit_oside_tags = []
    for i in data.index:
        tags = []
        q1_units = set(data.loc[i, 'question1'].lower().split())
        q2_units = set(data.loc[i, 'question2'].lower().split())
        for unit in punit_oside:
            if unit in q1_units and unit not in q2_units:
                tags.append(1.0)
            elif unit not in q1_units and unit in q2_units:
                tags.append(1.0)
            else:
                tags.append(0.0)
        punit_oside_tags.append(tags)
    return punit_oside, punit_oside_tags


def powerful_units_dside_rate(sorted_units_power, punit_dside, data):
    num_least = 300
    units_power = dict(sorted_units_power)
    punit_dside_rate = []
    for i in data.index:
        rate = 1.0
        q1_units = set(data.loc[i, 'question1'].lower().split())
        q2_units = set(data.loc[i, 'question2'].lower().split())
        share_units = list(q1_units.intersection(q2_units))
        for unit in share_units:
            if unit in punit_dside:
                rate *= (1.0 - units_power[unit][6])
        punit_dside_rate.append(1-rate)
    return punit_dside_rate


def powerful_units_oside_rate(sorted_units_power, punit_oside, data):
    num_least = 300
    units_power = dict(sorted_units_power)
    punits_oside_rate = []
    for i in data.index:
        rate = 1.0
        q1_units = set(data.loc[i, 'question1'].lower().split())
        q2_units = set(data.loc[i, 'question2'].lower().split())
        q1_diff = list(set(q1_units).difference(set(q2_units)))
        q2_diff = list(set(q2_units).difference(set(q1_units)))
        all_diff = set(q1_diff + q2_diff)
        for unit in all_diff:
            if unit in punit_oside:
                rate *= (1.0 - units_power[unit][4])
        punits_oside_rate.append(1-rate)
    return punits_oside_rate


def edit_distance(q1, q2):
    str1 = q1.split(' ')
    str2 = q2.split(' ')
    matrix = [[i+j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1] + 1, matrix[i-1][j-1]+d)
            if j > i > 1 and str1[i-1] == str2[j-2] and str1[i-2] == str2[j-1]:
                d = 0
                matrix[i][j] = min(matrix[i][j], matrix[i-2][j-2] + d)
    return matrix[len(str1)][len(str2)]


def get_edit_distance(data):
    q1_len = data['question1'].apply(lambda x: len(list(x.split(' ')))).values
    q2_len = data['question2'].apply(lambda x: len(list(x.split(' ')))).values

    dist = [edit_distance(data.loc[i, 'question1'], data.loc[i, 'question2']) / np.max([q1_len, q2_len], axis=0)[i] for i in data.index]
    return dist


def generate_split_chars():
    for mode in ['train', 'test']:
        df_temp = pd.read_csv('./data/noextension/' + mode + '.csv', encoding='utf-8', engine='python')
        question1 = df_temp.question1.apply(lambda x: ' '.join(list(x.replace(' ', ''))))
        question2 = df_temp.question2.apply(lambda x: ' '.join(list(x.replace(' ', ''))))
        df_corpus = pd.DataFrame({
            'question1': question1,
            'question2': question2,
        })
        df_corpus.to_csv('./data/noextension/' + mode + '_corpus_char.csv', index=False)

    for i in range(10):
        for mode in ['train', 'dev', 'test']:
            df_temp = pd.read_csv('./data/noextension/' + str(i) + '/' + mode + '.csv', encoding='utf-8', engine='python')
            question1 = df_temp.question1.apply(lambda x: ' '.join(list(x.replace(' ', ''))))
            question2 = df_temp.question2.apply(lambda x: ' '.join(list(x.replace(' ', ''))))

            if mode == 'train':
                df_corpus = pd.DataFrame({
                    'question1': question1,
                    'question2': question2,
                    'label': df_temp.label
                })
            else:
                df_corpus = pd.DataFrame({
                    'question1': question1,
                    'question2': question2,
                })
            df_corpus.to_csv('./data/noextension/' + str(i) + '/' + mode + '_corpus_char.csv', index=False)


def generate_split_words():
    for mode in ['train', 'test']:
        df_temp = pd.read_csv('./data/noextension/' + mode + '.csv', encoding='utf-8', engine='python')
        question1 = df_temp.question1.apply(lambda x: ' '.join(jieba.cut(x.replace(' ', ''))))
        question2 = df_temp.question2.apply(lambda x: ' '.join(jieba.cut(x.replace(' ', ''))))
        df_corpus = pd.DataFrame({
            'question1': question1,
            'question2': question2,
        })
        df_corpus.to_csv('./data/noextension/' + mode + '_corpus_word.csv', index=False)

    for i in range(10):
        for mode in ['train', 'dev', 'test']:
            df_temp = pd.read_csv('./data/noextension/' + str(i) + '/' + mode + '.csv', encoding='utf-8', engine='python')
            question1 = df_temp.question1.apply(lambda x: ' '.join(jieba.cut(x.replace(' ', ''))))
            question2 = df_temp.question2.apply(lambda x: ' '.join(jieba.cut(x.replace(' ', ''))))

            if mode == 'train':
                df_corpus = pd.DataFrame({
                    'question1': question1,
                    'question2': question2,
                    'label': df_temp.label
                })
            else:
                df_corpus = pd.DataFrame({
                    'question1': question1,
                    'question2': question2,
                })
            df_corpus.to_csv('./data/noextension/' + str(i) + '/' + mode + '_corpus_word.csv', index=False)


def generate_features_csv():
    # prepare and load THUOCL medical file
    with open('./data/THUOCL_medical.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    with open('./data/THUOCL_medical.txt', 'w', encoding='utf-8') as f:
        content = content.replace('\t', ' ')
        f.write(content)
    jieba.load_userdict('./data/THUOCL_medical.txt')

    print('*' * 10 + ' Generating chars corpus file ' + '*' * 10)
    generate_split_chars()
    print('*' * 10 + ' Generating words corpus file ' + '*' * 10)
    generate_split_words()

    all_train_data = pd.read_csv('./data/noextension/train_corpus_char.csv', encoding='utf-8', engine='python')
    corpus = list(all_train_data.question1) + list(all_train_data.question2)
    all_test_data = pd.read_csv('./data/noextension/test_corpus_char.csv', encoding='utf-8', engine='python')
    corpus += list(all_test_data.question1) + list(all_test_data.question2)
    vectorizer_char = TfidfVectorizer(token_pattern=r'[^\s]+').fit(corpus)

    all_train_data = pd.read_csv('./data/noextension/train_corpus_word.csv', encoding='utf-8', engine='python')
    corpus = list(all_train_data.question1) + list(all_train_data.question2)
    all_test_data = pd.read_csv('./data/noextension/test_corpus_word.csv', encoding='utf-8', engine='python')
    corpus += list(all_test_data.question1) + list(all_test_data.question2)
    vectorizer_word = TfidfVectorizer(token_pattern=r'[^\s]+').fit(corpus)

    print('*' * 10 + ' Generating feature file ' + '*' * 10)
    for i in tqdm(range(10)):
        sorted_chars_power = None
        sorted_words_power = None
        for mode in ['train', 'dev', 'test']:
            data = pd.read_csv('./data/noextension/' + str(i) + '/' + mode + '_corpus_char.csv', encoding='utf-8', engine='python')
            if mode == 'train':
                sorted_chars_power = generate_powerful_unit(data)

            len_diff_char = get_len_diff(data)
            edit_char = get_edit_distance(data)
            vocab, q1_tfidf, q2_tfidf = get_tfidf_vector(data, vectorizer_char)
            adjusted_common_char_ratio = adjust_common_units_ratio_by_tfidf(data, vocab, q1_tfidf, q2_tfidf)
            pchar_dside, pchar_dside_tags = powerful_units_dside_tag(sorted_chars_power, data, 1, 0.7)
            pchar_dside_rate = powerful_units_dside_rate(sorted_chars_power, pchar_dside, data)
            pchar_oside, pchar_oside_tags = powerful_units_oside_tag(sorted_chars_power, data, 1, 0.7)
            pchar_oside_rate = powerful_units_oside_rate(sorted_chars_power, pchar_oside, data)

            data = pd.read_csv('./data/noextension/' + str(i) + '/' + mode + '_corpus_word.csv', encoding='utf-8', engine='python')
            if mode == 'train':
                sorted_words_power = generate_powerful_unit(data)

            len_diff_word = get_len_diff(data)
            edit_word = get_edit_distance(data)
            vocab, q1_tfidf, q2_tfidf = get_tfidf_vector(data, vectorizer_word)
            adjusted_common_word_ratio = adjust_common_units_ratio_by_tfidf(data, vocab, q1_tfidf, q2_tfidf)
            pword_dside, pword_dside_tags = powerful_units_dside_tag(sorted_words_power, data, 1, 0.7)
            pword_dside_rate = powerful_units_dside_rate(sorted_words_power, pword_dside, data)
            pword_oside, pword_oside_tags = powerful_units_oside_tag(sorted_words_power, data, 1, 0.7)
            pword_oside_rate = powerful_units_oside_rate(sorted_words_power, pword_oside, data)

            df = pd.DataFrame({'len_diff_char': len_diff_char, 'edit_char': edit_char, 'len_diff_word': len_diff_word, 'edit_word': edit_word,
                               'adjusted_common_char_ratio': adjusted_common_char_ratio, 'adjusted_common_word_ratio': adjusted_common_word_ratio,
                               'pchar_dside_rate': pchar_dside_rate, 'pchar_oside_rate': pchar_oside_rate, 'pword_dside_rate': pword_dside_rate, 'pword_oside_rate': pword_oside_rate})
            df.to_csv('./data/noextension/' + str(i) + '/' + mode + '_feats.csv', index=False)


generate_features_csv()
