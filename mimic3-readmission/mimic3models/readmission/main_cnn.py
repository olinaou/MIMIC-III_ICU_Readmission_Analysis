import argparse
import os

import numpy as np

os.environ["KERAS_BACKEND"] = "tensorflow"

import importlib.machinery
import re
from mimic3benchmark.util import *
import matplotlib.pyplot as plt

from sklearn import metrics
from mimic3models.readmission import utils
from mimic3benchmark.readers import ReadmissionReader

from mimic3models.preprocessing import Discretizer, Normalizer
# from mimic3models import metrics
from mimic3models.metrics import print_metrics_binary
from mimic3models import keras_utils
from mimic3models import common_utils

from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from utilities.data_loader import get_embeddings
import statistics

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


def get_additional_features(filenames):
    stays_custom_model = dataframe_from_csv(os.path.join(args.additional_features_path, 'stays_all_drop_sampled.csv'),
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
parser.add_argument('--additional_features_path', type=str, default='../../scripts/')
args = parser.parse_args()
print(args)


def age_normalize(demographic, age_means, age_std):
    demographic = np.asmatrix(demographic)

    demographic[:, 0] = (demographic[:, 0] - age_means) / age_std
    return demographic.tolist()


def normalize_standard(df, columns):
    for col in columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    return df


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
feature_cols = ['LOS', 'Hos_LOS', 'Num_Prev_Hos_Adm']
additional_features = get_additional_features(names)
additional_features_list = normalize_standard(additional_features, feature_cols).values.tolist()

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
args_dict['input_dim'] = 393

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
train_raw = utils.load_train_data_model1(train_reader, discretizer, normalizer, diseases_embedding, demographic,
                                         additional_features_list, args.small_part)
np.nan_to_num(train_raw[0], copy=False)

print('train_raw: ', len(train_raw[0]))

print('train_raw train_raw[0][0]: ', len(train_raw[0][0]))
print('train_raw train_raw[0][1]: ', len(train_raw[0][1]))

# print('train_raw: ', len(train_raw[0][0][0]))

N1 = val_reader.get_number_of_examples()
ret1 = common_utils.read_chunk(val_reader, N1)

names1 = ret1["name"]

additional_features = get_additional_features(names1)
additional_features_list = normalize_standard(additional_features, feature_cols).values.tolist()

diseases_list1 = get_diseases(names1, dataset_subject_dir)
diseases_embedding1 = disease_embedding(embeddings, word_indices, diseases_list1)
demographic1 = get_demographic(names1, dataset_subject_dir)
demographic1 = age_normalize(demographic1, age_means, age_std)

val_raw = utils.load_data_model1(val_reader, discretizer, normalizer, diseases_embedding1, demographic1,
                                 additional_features_list, args.small_part)
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
    additional_features = get_additional_features(names_t)
    additional_features_list = normalize_standard(additional_features, feature_cols).values.tolist()

    diseases_list_t = get_diseases(names_t, dataset_subject_dir)
    diseases_embedding_t = disease_embedding(embeddings, word_indices, diseases_list_t)
    demographic_t = get_demographic(names_t, dataset_subject_dir)
    demographic_t = age_normalize(demographic_t, age_means, age_std)

    ret = utils.load_data_model1(test_reader, discretizer, normalizer, diseases_embedding_t, demographic_t,
                                 additional_features_list, args.small_part, return_names=True)

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
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k')

plt.xlim([0., 1.])
plt.ylim([0., 1.])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")

fig.savefig('ROC_' + args.load_state + '.png')

plt.show()
