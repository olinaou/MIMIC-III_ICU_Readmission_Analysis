import os

import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt

from mimic3benchmark.util import *


def read_diagnose(subject_path, icustay):
    diagnoses = dataframe_from_csv(os.path.join(subject_path, 'diagnoses.csv'), index_col=None)
    diagnoses = diagnoses.ix[(diagnoses.ICUSTAY_ID == int(icustay))]
    diagnoses = diagnoses['ICD9_CODE'].values.tolist()

    return diagnoses


def read_diagnoses_features(path):
    diagnoses = pd.read_csv(os.path.join(path, 'diagnoses.csv'), index_col=None, nrows=1)
    return diagnoses.columns


def get_diseases(names, path):
    disease_list = []
    namelist = []
    for element in names:
        x = element.split('_')
        namelist.append((x[0], x[1]))
    for x in namelist:
        subject = x[0]
        icustay = x[1]
        subject_path = os.path.join(path, subject)
        disease = read_diagnose(subject_path, icustay)
        disease_list.append(disease)
    return disease_list


def disease_embedding(embeddings, word_indices, diseases_list):
    emb_list = []
    for diseases in diseases_list:
        emb_period = [0] * 300
        skip = 0
        for disease in diseases:
            k = 'IDX_' + str(disease)
            if k not in word_indices.keys():
                skip += 1
                continue
            index = word_indices[k]
            emb_disease = embeddings[index]
            emb_period = [sum(x) for x in zip(emb_period, emb_disease)]
        emb_list.append(emb_period)
    return emb_list


def find_majority(k):
    myMap = {}
    maximum = ('', 0)
    for n in k:
        if n in myMap:
            myMap[n] += 1
        else:
            myMap[n] = 1

        if myMap[n] > maximum[1]: maximum = (n, myMap[n])

    return maximum


def logit(X, cont_channels, begin_pos, end_pos):
    X = np.asmatrix(X)

    index = list(range(59))
    no = list(range(59, 76))
    idx_features = []
    features = []
    majority_index = list(set(index) - set(cont_channels))
    reg_index = list(set(cont_channels) - set(no))

    for idx in majority_index:
        begine = 0
        end = 0
        for i, item in enumerate(begin_pos):
            if item == idx:
                begin = idx
                end = end_pos[i]

                flat_list = [map(int, my_lst) for my_lst in X[:, begin:end].tolist()]

                flat_list = [(''.join(map(str, my_lst))) for my_lst in flat_list]

                value = find_majority(flat_list)[0]
                for ch in list(value):
                    idx_features.append(float(ch))
    for idx in reg_index:
        regr = linear_model.LinearRegression()
        flat_list = [item for sublist in X[:, idx].tolist() for item in sublist]

        time = [[i] for i in range(1, len(flat_list) + 1)]

        regr.fit(time, flat_list)
        a = regr.coef_[0]
        b = regr.intercept_
        features.append(a)
        features.append(b)
    return idx_features, features


def column_sum(M):
    s = M.shape[0]
    column_sums = [sum([row[i] for row in M]) for i in range(0, len(M[0]))]
    newList = [x / s for x in column_sums]
    return newList


def get_additional_features(base, filenames):
    stays_custom_model = dataframe_from_csv(os.path.join(base, 'stays_all_drop_sampled.csv'),
                                            index_col='ICUSTAY_ID')
    features = []
    namelist = []
    cols = ['LOS', 'Hos_LOS', 'Num_Prev_Hos_Adm']
    df = pd.DataFrame(columns=cols);
    for element in filenames:
        x = element.split('_')
        namelist.append((x[0], x[1], x[2]))
    for x in namelist:
        icu_stay = int(x[1])

        df = df.append(stays_custom_model.loc[icu_stay][cols], ignore_index=True)
        # current_features = stays_custom_model.loc[icu_stay][cols].tolist()
        # features.append(current_features)
    return df


def svm_f_importance(coef, names):
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()
