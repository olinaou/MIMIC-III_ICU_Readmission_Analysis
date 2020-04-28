import argparse
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import importlib.machinery
import re

from mimic3models.readmission import utils

from mimic3models.preprocessing import Normalizer
# from mimic3models import metrics
from mimic3models import keras_utils

from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
import statistics

import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from mimic3benchmark.readers import ReadmissionReader
from mimic3benchmark.util import *
from mimic3models import common_utils
from mimic3models.metrics import print_metrics_binary
from mimic3models.preprocessing import Discretizer
from mimic3models.readmission_baselines.utils import save_results
from utilities.data_loader import get_embeddings

g_map = {'F': 1, 'M': 2}

e_map = {'ASIAN': 1,
         'BLACK': 2,
         'HISPANIC': 3,
         'WHITE': 4,
         'OTHER': 5,  # map everything else to 5 (OTHER)
         'UNABLE TO OBTAIN': 0,
         'PATIENT DECLINED TO ANSWER': 0,
         'UNKNOWN': 0,
         '': 0}

i_map = {'Government': 0,
         'Self Pay': 1,
         'Medicare': 2,
         'Private': 3,
         'Medicaid': 4}


def read_diagnose(subject_path, icustay):
    diagnoses = dataframe_from_csv(os.path.join(subject_path, 'diagnoses.csv'), index_col=None)
    diagnoses = diagnoses.ix[(diagnoses.ICUSTAY_ID == int(icustay))]
    diagnoses = diagnoses['ICD9_CODE'].values.tolist()

    return diagnoses


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


def read_demographic(subject_path, icustay, episode):
    demographic_re = [0] * 14
    demographic = dataframe_from_csv(os.path.join(subject_path, episode + '_readmission.csv'), index_col=None)
    age_start = 0
    gender_start = 1
    enhnicity_strat = 3
    insurance_strat = 9
    demographic_re[age_start] = float(demographic['Age'].iloc[0])
    demographic_re[gender_start - 1 + int(demographic['Gender'].iloc[0])] = 1
    demographic_re[enhnicity_strat + int(demographic['Ethnicity'].iloc[0])] = 1
    insurance = dataframe_from_csv(os.path.join(subject_path, 'stays_readmission.csv'), index_col=None)

    insurance = insurance.ix[(insurance.ICUSTAY_ID == int(icustay))]

    demographic_re[insurance_strat + i_map[insurance['INSURANCE'].iloc[0]]] = 1

    return demographic_re


def get_demographic(names, path):
    demographic_list = []
    namelist = []
    for element in names:
        x = element.split('_')
        namelist.append((x[0], x[1], x[2]))
    for x in namelist:
        subject = x[0]
        icustay = x[1]
        episode = x[2]

        subject_path = os.path.join(path, subject)
        demographic = read_demographic(subject_path, icustay, episode)
        demographic_list.append(demographic)
    return demographic_list


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


fig = plt.figure(figsize=(7, 7))

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
args = parser.parse_args()
print(args)


def age_normalize(demographic, age_means, age_std):
    demographic = np.asmatrix(demographic)

    demographic[:, 0] = (demographic[:, 0] - age_means) / age_std
    return demographic.tolist()


if args.small_part:
    args.save_every = 2 ** 30

base = "../../scripts/"
subject_data = "output_subjects_1000"
episodes_data = "output_episodes_1000"
listfiles_data = "output_listfiles_1000"
dataset_subject_dir = base + subject_data + "/"
dataset_episode_dir = base + episodes_data + "/"
listfiles_train_dir0 = base + listfiles_data + "/0_train_listfile801010.csv"
listfiles_val_dir0 = base + listfiles_data + "/0_val_listfile801010.csv"
listfiles_test_dir0 = base + listfiles_data + "/0_test_listfile801010.csv"

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')
# Read embedding
embeddings, word_indices = get_embeddings(corpus='claims_codes_hs', dim=300)

train_reader = ReadmissionReader(dataset_dir=dataset_episode_dir,
                                 listfile=listfiles_train_dir0)

val_reader = ReadmissionReader(dataset_dir=dataset_episode_dir,
                               listfile=listfiles_val_dir0)

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          imput_strategy='previous',
                          start_time='zero')

N = train_reader.get_number_of_examples()
ret = common_utils.read_chunk(train_reader, N)
data = ret["X"]
ts = ret["t"]
labels = ret["y"]
names = ret["name"]
diseases_list = get_diseases(names, dataset_subject_dir)
diseases_embedding = disease_embedding(embeddings, word_indices, diseases_list)
demographic = get_demographic(names, dataset_subject_dir)

age_means = sum(demographic[:][0])
age_std = statistics.stdev(demographic[:][0])

print('age_means: ', age_means)
print('age_std: ', age_std)
demographic = age_normalize(demographic, age_means, age_std)

discretizer_header = discretizer.transform(ret["X"][0])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
normalizer = Normalizer(fields=cont_channels)  # choose here onlycont vs all

data = [discretizer.transform_end_t_hours(X, los=t)[0] for (X, t) in zip(data, ts)]

[normalizer._feed_data(x=X) for X in data]
normalizer._use_params()

args_dict = dict(args._get_kwargs())

args_dict['task'] = 'ihm'
args_dict['target_repl'] = target_repl

# Build the model
print("==> using model {}".format(args.network))
print('os.path.basename(args.network), args.network: ', os.path.basename(args.network), args.network)
model_module = importlib.machinery.SourceFileLoader(os.path.basename(args.network), args.network).load_module()
model = model_module.Network(**args_dict)
suffix = ".bs{}{}{}.ts{}{}".format(args.batch_size,
                                   ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                   ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                   args.timestep,
                                   ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "")
model.final_name = args.prefix + model.say_name() + suffix
print("==> model.final_name:", model.final_name)

# Compile the model
print("==> compiling the model")

# NOTE: one can use binary_crossentropy even for (B, T, C) shape.
#       It will calculate binary_crossentropies for each class
#       and then take the mean over axis=-1. Tre results is (B, T).

loss = 'binary_crossentropy'
loss_weights = None
print(model)
model.compile(optimizer=Adam(lr=0.001, beta_1=0.9), loss=loss, loss_weights=loss_weights)

model.summary()

# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))

# Read data
train_raw = utils.load_train_data(train_reader, discretizer, normalizer, diseases_embedding, demographic,
                                  args.small_part)
np.nan_to_num(train_raw[0], copy=False)

print('train_raw: ', len(train_raw[0]))

print('train_raw train_raw[0][0]: ', len(train_raw[0][0]))
print('train_raw train_raw[0][1]: ', len(train_raw[0][1]))

# print('train_raw: ', len(train_raw[0][0][0]))

N1 = val_reader.get_number_of_examples()
ret1 = common_utils.read_chunk(val_reader, N1)

names1 = ret1["name"]
diseases_list1 = get_diseases(names1, dataset_subject_dir)
diseases_embedding1 = disease_embedding(embeddings, word_indices, diseases_list1)
demographic1 = get_demographic(names1, dataset_subject_dir)
demographic1 = age_normalize(demographic1, age_means, age_std)

val_raw = utils.load_data(val_reader, discretizer, normalizer, diseases_embedding1, demographic1, args.small_part)
np.nan_to_num(val_raw[0], copy=False)
if target_repl:
    T = train_raw[0][0].shape[0]


    def extend_labels(data):
        data = list(data)
        labels = np.array(data[1])  # (B,)
        data[1] = [labels, None]
        data[1][1] = np.expand_dims(labels, axis=-1).repeat(T, axis=1)  # (B, T)
        data[1][1] = np.expand_dims(data[1][1], axis=-1)  # (B, T, 1)
        return data


    train_raw = extend_labels(train_raw)
    val_raw = extend_labels(val_raw)

if args.mode == 'train':

    # Prepare training
    path = 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state'

    metrics_callback = keras_utils.ReadmissionMetrics(train_data=train_raw,
                                                      val_data=val_raw,
                                                      target_repl=(args.target_repl_coef > 0),
                                                      batch_size=args.batch_size,
                                                      verbose=args.verbose)
    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)

    if not os.path.exists('keras_logs'):
        os.makedirs('keras_logs')
    csv_logger = CSVLogger(os.path.join('keras_logs', model.final_name + '.csv'),
                           append=True, separator=';')

    print("==> training")
    model.fit(x=train_raw[0],
              y=train_raw[1],
              validation_data=val_raw,
              nb_epoch=n_trained_chunks + args.epochs,
              initial_epoch=n_trained_chunks,
              callbacks=[metrics_callback, saver, csv_logger],
              shuffle=True,
              verbose=args.verbose,
              batch_size=args.batch_size)

elif args.mode == 'test':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw

    test_reader = ReadmissionReader(dataset_dir=dataset_episode_dir,
                                    listfile=listfiles_test_dir0)

    N = test_reader.get_number_of_examples()
    re = common_utils.read_chunk(test_reader, N)

    names_t = re["name"]
    diseases_list_t = get_diseases(names_t, dataset_subject_dir)
    diseases_embedding_t = disease_embedding(embeddings, word_indices, diseases_list_t)
    demographic_t = get_demographic(names_t, dataset_subject_dir)
    demographic_t = age_normalize(demographic_t, age_means, age_std)

    ret = utils.load_data(test_reader, discretizer, normalizer, diseases_embedding_t, demographic_t, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]
    np.nan_to_num(data, copy=False)

    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions_plt = predictions
    predictions = np.array(predictions)[:, 0]
    print_metrics_binary(labels, predictions)

    predictions_plt2 = np.array(predictions_plt[:, 0])
    if len(predictions_plt2.shape) == 1:
        predictions_plt2 = np.stack([1 - predictions_plt2, predictions_plt2]).transpose((1, 0))

    fpr, tpr, thresh = metrics.roc_curve(labels, predictions_plt2[:, 1])
    auc = metrics.roc_auc_score(labels, predictions_plt2[:, 1])
    plt.plot(fpr, tpr, lw=2, label="CNN= %0.3f auc" % auc)

    path = os.path.join("test_predictions", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)

else:
    raise ValueError("Wrong value for args.mode")

# =============================
# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k')
#
# plt.xlim([0., 1.])
# plt.ylim([0., 1.])
#
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC curve')
# plt.legend(loc="lower right")
#
# fig.savefig('ROC0.png')
#
# plt.show()

#fig = plt.figure(figsize=(7, 7))


def read_diagnose(subject_path, icustay):
    diagnoses = dataframe_from_csv(os.path.join(subject_path, 'diagnoses.csv'), index_col=None)
    diagnoses = diagnoses.ix[(diagnoses.ICUSTAY_ID == int(icustay))]
    diagnoses = diagnoses['ICD9_CODE'].values.tolist()

    return diagnoses


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

#base = "../../../scripts/"
subject_data = "output_subjects_1000"
episodes_data = "output_episodes_1000"
listfiles_data = "output_listfiles_1000"
dataset_subject_dir = base + subject_data + "/"
dataset_episode_dir = base + episodes_data + "/"
listfiles_train_dir0 = base + listfiles_data + "/0_train_listfile801010.csv"
listfiles_val_dir0 = base + listfiles_data + "/0_val_listfile801010.csv"
listfiles_test_dir0 = base + listfiles_data + "/0_test_listfile801010.csv"

embeddings, word_indices = get_embeddings(corpus='claims_codes_hs', dim=300)

# Build readers, discretizers, normalizers
train_reader = ReadmissionReader(dataset_dir=dataset_episode_dir,
                                 listfile=listfiles_train_dir0)

val_reader = ReadmissionReader(dataset_dir=dataset_episode_dir,
                               listfile=listfiles_val_dir0)

test_reader = ReadmissionReader(dataset_dir=dataset_episode_dir,
                                listfile=listfiles_test_dir0)

discretizer = Discretizer(timestep=float(1.0),
                          store_masks=True,
                          imput_strategy='previous',
                          start_time='zero')

N = train_reader.get_number_of_examples()
ret = common_utils.read_chunk(train_reader, N)
data = ret["X"]
ts = ret["t"]
train_y = ret["y"]
train_names = ret["name"]
diseases_list = get_diseases(train_names, dataset_subject_dir)
diseases_embedding = disease_embedding(embeddings, word_indices, diseases_list)

d, discretizer_header, begin_pos, end_pos = discretizer.transform_reg(data[0])

discretizer_header = discretizer_header.split(',')

cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

da = [discretizer.transform_end_t_hours_reg(X, los=t)[1] for (X, t) in zip(data, ts)]
mask = [column_sum(x) for x in da]

# train_set=[]
d = [discretizer.transform_end_t_hours_reg(X, los=t)[0] for (X, t) in zip(data, ts)]

idx_features_train = [logit(X, cont_channels, begin_pos, end_pos)[0] for X in d]
features_train = [logit(X, cont_channels, begin_pos, end_pos)[1] for X in d]

train_X = features_train

scaler = MinMaxScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)

train_X = [np.hstack([X, d]) for (X, d) in zip(train_X, diseases_embedding)]

train_X = [np.hstack([X, d]) for (X, d) in zip(train_X, idx_features_train)]

train_X = [np.hstack([X, d]) for (X, d) in zip(train_X, mask)]

labels_1 = []
labels_0 = []
data_1 = []
data_0 = []
for i in range(len(train_y)):
    if train_y[i] == 1:
        labels_1.append(train_y[i])
        data_1.append(train_X[i])
    elif train_y[i] == 0:
        labels_0.append(train_y[i])
        data_0.append(train_X[i])

print('labels_1:', len(labels_1))
print('labels_0:', len(labels_0))
indices = np.random.choice(len(labels_0), len(labels_1), replace=False)
labels_0_sample = [labels_0[idx] for idx in indices]
# print('labels_0_sample: ', labels_0_sample)
print('len(labels_0_sample): ', len(labels_0_sample))

data_0_sample = [data_0[idx] for idx in indices]
# print('data_0_sample: ', data_0_sample)
print('len(data_0_sample): ', len(data_0_sample))

data_new = data_0_sample + data_1
label_new = labels_0_sample + labels_1

c = list(zip(data_new, label_new))

random.shuffle(c)

data_new, label_new = zip(*c)
train_X = list(data_new)
train_y = list(label_new)
# print('data_new: ', data_new)
print('data_new: ', len(train_X))
# print('label_new: ', label_new)
print('label_new: ', len(train_y))

# -------------------------

N_val = val_reader.get_number_of_examples()
ret_val = common_utils.read_chunk(val_reader, N_val)
data_val = ret_val["X"]
ts_val = ret_val["t"]
val_y = ret_val["y"]
val_names = ret_val["name"]

diseases_list_val = get_diseases(val_names, dataset_subject_dir)
diseases_embedding_val = disease_embedding(embeddings, word_indices, diseases_list_val)

# ----------
da_val = [discretizer.transform_end_t_hours_reg(X, los=t)[1] for (X, t) in zip(data_val, ts_val)]
mask_val = [column_sum(x) for x in da_val]

# train_set=[]
d_val = [discretizer.transform_end_t_hours_reg(X, los=t)[0] for (X, t) in zip(data_val, ts_val)]

# ---------
# val_set=[]
idx_features_val = [logit(X, cont_channels, begin_pos, end_pos)[0] for X in d_val]
features_val = [logit(X, cont_channels, begin_pos, end_pos)[1] for X in d_val]

val_X = scaler.transform(features_val)

val_X = [np.hstack([X, d]) for (X, d) in zip(val_X, diseases_embedding_val)]
val_X = [np.hstack([X, d]) for (X, d) in zip(val_X, idx_features_val)]
val_X = [np.hstack([X, d]) for (X, d) in zip(val_X, mask_val)]

# -------------------------

N_test = test_reader.get_number_of_examples()
ret_test = common_utils.read_chunk(test_reader, N_test)
data_test = ret_test["X"]
ts_test = ret_test["t"]
test_y = ret_test["y"]
test_names = ret_test["name"]

diseases_list_test = get_diseases(test_names, dataset_subject_dir)
diseases_embedding_test = disease_embedding(embeddings, word_indices, diseases_list_test)

# ----------
da_test = [discretizer.transform_end_t_hours_reg(X, los=t)[1] for (X, t) in zip(data_test, ts_test)]
mask_test = [column_sum(x) for x in da_test]

# train_set=[]
d_test = [discretizer.transform_end_t_hours_reg(X, los=t)[0] for (X, t) in zip(data_test, ts_test)]
# ----------

# data_test=[]
idx_features_test = [logit(X, cont_channels, begin_pos, end_pos)[0] for X in d_test]
features_test = [logit(X, cont_channels, begin_pos, end_pos)[1] for X in d_test]

test_X = scaler.transform(features_test)

test_X = [np.hstack([X, d]) for (X, d) in zip(test_X, diseases_embedding_test)]
test_X = [np.hstack([X, d]) for (X, d) in zip(test_X, idx_features_test)]
test_X = [np.hstack([X, d]) for (X, d) in zip(test_X, mask_test)]

# =========SVM====================
penalty = ('l2')
# file_name = '{}.{}.{}.C{}'.format(penalty, 0.001)


logreg = SVC(probability=True)
logreg.fit(train_X, train_y)

# -----------------
common_utils.create_directory('svm_results')
common_utils.create_directory('svm_predictions')

with open(os.path.join('svm_results', 'train.json'), 'w') as res_file:
    ret = print_metrics_binary(train_y, logreg.predict_proba(train_X))
    ret = {k: float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

with open(os.path.join('svm_results', 'val.json'), 'w') as res_file:
    ret = print_metrics_binary(val_y, logreg.predict_proba(val_X))
    ret = {k: float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

prediction = logreg.predict_proba(test_X)[:, 1]

with open(os.path.join('svm_results', 'test.json'), 'w') as res_file:
    ret = print_metrics_binary(test_y, prediction)
    ret = {k: float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

    predictions = np.array(prediction)
    if (len(predictions.shape) == 1):
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    fpr, tpr, thresh = metrics.roc_curve(test_y, predictions[:, 1])
    auc = metrics.roc_auc_score(test_y, predictions[:, 1])
    plt.plot(fpr, tpr, lw=2, label="SVM= %0.3f" % auc)

save_results(test_names, prediction, test_y, os.path.join('svm_predictions', 'svm.csv'))

# =============LR================


logreg = LogisticRegression(penalty=penalty, C=0.001, random_state=42)
logreg.fit(train_X, train_y)

# -----------------
common_utils.create_directory('lr_results')
common_utils.create_directory('lr_predictions')

with open(os.path.join('lr_results', 'train.json'), 'w') as res_file:
    ret = print_metrics_binary(train_y, logreg.predict_proba(train_X))
    ret = {k: float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

with open(os.path.join('lr_results', 'val.json'), 'w') as res_file:
    ret = print_metrics_binary(val_y, logreg.predict_proba(val_X))
    ret = {k: float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

prediction = logreg.predict_proba(test_X)[:, 1]

with open(os.path.join('lr_results', 'test.json'), 'w') as res_file:
    ret = print_metrics_binary(test_y, prediction)
    ret = {k: float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

    predictions = np.array(prediction)
    if (len(predictions.shape) == 1):
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    fpr, tpr, thresh = metrics.roc_curve(test_y, predictions[:, 1])
    auc = metrics.roc_auc_score(test_y, predictions[:, 1])
    plt.plot(fpr, tpr, lw=2, label="LR= %0.3f" % auc)

save_results(test_names, prediction, test_y, os.path.join('lr_predictions', 'lr.csv'))

# ===========RF==================


logreg = RandomForestClassifier(oob_score=True, max_depth=50, random_state=0)
logreg.fit(train_X, train_y)

# -----------------
common_utils.create_directory('rf_results')
common_utils.create_directory('rf_predictions')

with open(os.path.join('rf_results', 'train.json'), 'w') as res_file:
    ret = print_metrics_binary(train_y, logreg.predict_proba(train_X))
    ret = {k: float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

with open(os.path.join('rf_results', 'val.json'), 'w') as res_file:
    ret = print_metrics_binary(val_y, logreg.predict_proba(val_X))
    ret = {k: float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

prediction = logreg.predict_proba(test_X)[:, 1]

with open(os.path.join('rf_results', 'test.json'), 'w') as res_file:
    ret = print_metrics_binary(test_y, prediction)
    ret = {k: float(v) for k, v in ret.items()}
    json.dump(ret, res_file)

    predictions = np.array(prediction)
    if (len(predictions.shape) == 1):
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    fpr, tpr, thresh = metrics.roc_curve(test_y, predictions[:, 1])
    auc = metrics.roc_auc_score(test_y, predictions[:, 1])
    plt.plot(fpr, tpr, lw=2, label="RF= %0.3f" % auc)

save_results(test_names, prediction, test_y, os.path.join('rf_predictions', 'rf.csv'))

# =============NB================


# logreg = GaussianNB()
# logreg.fit(train_X, train_y)
#
# # -----------------
# common_utils.create_directory('nb_results')
# common_utils.create_directory('nb_predictions')
#
# with open(os.path.join('nb_results', 'train.json'), 'w') as res_file:
#     ret = print_metrics_binary(train_y, logreg.predict_proba(train_X))
#     ret = {k: float(v) for k, v in ret.items()}
#     json.dump(ret, res_file)
#
# with open(os.path.join('nb_results', 'val.json'), 'w') as res_file:
#     ret = print_metrics_binary(val_y, logreg.predict_proba(val_X))
#     ret = {k: float(v) for k, v in ret.items()}
#     json.dump(ret, res_file)
#
# prediction = logreg.predict_proba(test_X)[:, 1]
#
# with open(os.path.join('nb_results', 'test.json'), 'w') as res_file:
#     ret = print_metrics_binary(test_y, prediction)
#     ret = {k: float(v) for k, v in ret.items()}
#     json.dump(ret, res_file)
#
#     predictions = np.array(prediction)
#     if (len(predictions.shape) == 1):
#         predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))
#
#     fpr, tpr, thresh = metrics.roc_curve(test_y, predictions[:, 1])
#     auc = metrics.roc_auc_score(test_y, predictions[:, 1])
#     plt.plot(fpr, tpr, lw=2, label="NB= %0.3f" % auc)
#
# save_results(test_names, prediction, test_y, os.path.join('nb_predictions', 'nb.csv'))

# =============================
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k')

plt.xlim([0., 1.])
plt.ylim([0., 1.])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")

fig.savefig('ROC0.png')

plt.show()

