import csv
import os
import sys

import numpy as np
from dateutil.relativedelta import relativedelta

from mimic3benchmark.util import *


def read_patients_table(mimic3_path):
    pats = dataframe_from_csv(os.path.join(mimic3_path, 'PATIENTS.csv'))
    pats = pats[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']]
    pats.DOB = pd.to_datetime(pats.DOB)
    pats.DOD = pd.to_datetime(pats.DOD)
    return pats


def read_patients_table_with_expire_flag(mimic3_path):
    pats = dataframe_from_csv(os.path.join(mimic3_path, 'PATIENTS.csv'))
    pats = pats[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'EXPIRE_FLAG']]
    pats.DOB = pd.to_datetime(pats.DOB)
    pats.DOD = pd.to_datetime(pats.DOD)
    return pats


def read_admissions_table(mimic3_path):
    admits = dataframe_from_csv(os.path.join(mimic3_path, 'ADMISSIONS.csv'))
    admits = admits[
        ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'INSURANCE', 'RELIGION', 'MARITAL_STATUS',
         'ETHNICITY', 'DIAGNOSIS']]
    admits.ADMITTIME = pd.to_datetime(admits.ADMITTIME)
    admits.DISCHTIME = pd.to_datetime(admits.DISCHTIME)
    admits.DEATHTIME = pd.to_datetime(admits.DEATHTIME)
    return admits


def read_icustays_table(mimic3_path):
    stays = dataframe_from_csv(os.path.join(mimic3_path, 'ICUSTAYS.csv'))
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    return stays


def read_transfers_table(mimic3_path):
    transfers = dataframe_from_csv(os.path.join(mimic3_path, 'TRANSFERS.csv'))
    transfers.INTIME = pd.to_datetime(transfers.INTIME)
    transfers.OUTTIME = pd.to_datetime(transfers.OUTTIME)

    transfersnotnull = transfers.loc[transfers.ICUSTAY_ID.notnull()]
    # print(transfersnotnull)
    transfersisnull = transfers.loc[transfers.ICUSTAY_ID.isnull()]

    transfersnotnull = transfersnotnull.drop_duplicates('ICUSTAY_ID', keep='last')
    # print(transfersnotnull)
    transfers = pd.concat([transfersnotnull, transfersisnull])
    return transfers


def read_icd_diagnoses_table(mimic3_path):
    codes = dataframe_from_csv(os.path.join(mimic3_path, 'D_ICD_DIAGNOSES.csv'))
    codes = codes[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']]
    diagnoses = dataframe_from_csv(os.path.join(mimic3_path, 'DIAGNOSES_ICD.csv'))
    diagnoses = diagnoses.merge(codes, how='inner', left_on='ICD9_CODE', right_on='ICD9_CODE')
    diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']] = diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']].astype(int)
    return diagnoses


# =======================================


def read_icd_procedures_table(mimic3_path):
    codes = dataframe_from_csv(os.path.join(mimic3_path, 'D_ICD_PROCEDURES.csv'))
    codes = codes[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']]
    diagnoses = dataframe_from_csv(os.path.join(mimic3_path, 'PROCEDURES_ICD.csv'))
    diagnoses = diagnoses.merge(codes, how='inner', left_on='ICD9_CODE', right_on='ICD9_CODE')
    diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']] = diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']].astype(int)
    return diagnoses


def read_prescriptions_table(mimic3_path):
    prescriptions = dataframe_from_csv(os.path.join(mimic3_path, 'PRESCRIPTIONS.csv'))
    prescriptions.STARTDATE = pd.to_datetime(prescriptions.STARTDATE)
    prescriptions.ENDDATE = pd.to_datetime(prescriptions.ENDDATE)

    prescriptions = prescriptions.loc[prescriptions.ICUSTAY_ID.notnull()]
    prescriptions['ICUSTAY_ID'] = prescriptions['ICUSTAY_ID'].astype(int)
    prescriptions = prescriptions.loc[prescriptions.NDC != 0]

    # prescriptions=prescriptions.ICUSTAY_ID.notnull()&(prescriptions.ndc!=0)

    prescriptions = prescriptions[
        ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'NDC', 'DOSE_VAL_RX', 'DOSE_UNIT_RX', 'STARTDATE', 'ENDDATE']]

    # exclude = ['GSN']
    # prescriptions=prescriptions.ix[:, prescriptions.columns.difference(exclude)].hist()
    # print (prescriptions)
    return prescriptions


def merge_on_subject_admission_icustay(table1, table2):
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'],
                        right_on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'])


# =======================================
def read_events_table_by_row(mimic3_path, table):
    nb_rows = {'chartevents': 330712484, 'labevents': 27854056, 'outputevents': 4349219, 'inputevents_cv': 17527936,
               'inputevents_mv': 3618992}
    reader = csv.DictReader(open(os.path.join(mimic3_path, table.upper() + '.csv'), 'r'))
    for i, row in enumerate(reader):
        if 'ICUSTAY_ID' not in row:
            row['ICUSTAY_ID'] = ''
        yield row, i, nb_rows[table.lower()]


def count_icd_codes(diagnoses, output_path=None):
    codes = diagnoses[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']].drop_duplicates().set_index('ICD9_CODE')
    counts = diagnoses[['ICD9_CODE', 'HADM_ID']].drop_duplicates()
    codes['COUNT'] = counts.groupby('ICD9_CODE')['HADM_ID'].count()
    codes.COUNT = codes.COUNT.fillna(0).astype(int)
    codes = codes.ix[codes.COUNT > 0]
    if output_path:
        codes.to_csv(output_path, index_label='ICD9_CODE')
    return codes.sort_values('COUNT', ascending=False).reset_index()


def merge_on_subject(table1, table2):
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID'], right_on=['SUBJECT_ID'])


def merge_on_subject_admission(table1, table2):
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])


def get_age(start, end):
    if pd.isnull(start) or pd.isnull(end):
        return -1
    diff = relativedelta(end, start)
    age = diff.years + diff.months / 12 + diff.days / 365 + diff.hours / (365 * 24) + diff.minutes / (
            365 * 24 * 60) + diff.seconds / (365 * 24 * 60 * 60)
    return age


def add_age_to_icustays(stays):
    no_outliers = stays.DOB.between(stays.DOB.quantile(0.05), stays.DOB.quantile(0.999))
    index_names = stays[~no_outliers].index
    stays.drop(index_names, inplace=True)
    # stays['AGE'] = (stays.INTIME - stays.DOB).apply(lambda s: s / np.timedelta64(1, 's')) / 60. / 60 / 24 / 365
    stays['AGE'] = stays.apply(lambda row: get_age(row['DOB'], row['INTIME']), axis=1)
    stays.ix[stays.AGE < 0, 'AGE'] = 90
    return stays


def add_inhospital_mortality_to_icustays(stays):
    mortality = stays.DEATHTIME.notnull() & (
            (stays.ADMITTIME <= stays.DEATHTIME) & (stays.DISCHTIME >= stays.DEATHTIME))
    mortality = mortality | (stays.DEATHTIME.isnull() & stays.DOD.notnull() & (
            (stays.ADMITTIME <= stays.DOD) & (stays.DISCHTIME >= stays.DOD)))
    stays['MORTALITY'] = mortality.astype(int)
    stays['MORTALITY_INHOSPITAL'] = stays['MORTALITY']
    return stays


def add_inunit_mortality_to_icustays(stays):
    mortality = stays.DEATHTIME.notnull() & ((stays.INTIME <= stays.DEATHTIME) & (stays.OUTTIME >= stays.DEATHTIME))
    mortality = mortality | (stays.DEATHTIME.isnull() & stays.DOD.notnull() & (
            (stays.INTIME <= stays.DOD) & (stays.OUTTIME >= stays.DOD)))

    stays['MORTALITY_INUNIT'] = mortality.astype(int)
    return stays


def filter_icustays_on_age(stays, min_age=18, max_age=np.inf):
    stays = stays.loc[(stays.AGE >= min_age) & (stays.AGE <= max_age)]
    return stays


def filter_diagnoses_on_stays(diagnoses, stays):
    return diagnoses.merge(stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']], how='inner',
                           left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])


def break_up_stays_by_subject(stays, output_path, subjects=None, verbose=1):
    subjects = stays.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for i, subject_id in enumerate(subjects):
        if verbose:
            sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i + 1, nb_subjects))
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        stays.ix[stays.SUBJECT_ID == subject_id].sort_values(by='INTIME').to_csv(os.path.join(dn, 'stays.csv'),
                                                                                 index=False)

    if verbose:
        sys.stdout.write('DONE!\n')


def break_up_transfers_by_subject(transfers, output_path, subjects=None, verbose=1):
    subjects = transfers.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for i, subject_id in enumerate(subjects):
        if verbose:
            sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i + 1, nb_subjects))
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        transfers.ix[transfers.SUBJECT_ID == subject_id].sort_values(by='INTIME').to_csv(
            os.path.join(dn, 'transfers.csv'), index=False)
    if verbose:
        sys.stdout.write('DONE!\n')


def break_up_diagnoses_by_subject(diagnoses, output_path, subjects=None, verbose=1):
    subjects = diagnoses.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for i, subject_id in enumerate(subjects):
        if verbose:
            sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i + 1, nb_subjects))
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        diagnoses.ix[diagnoses.SUBJECT_ID == subject_id].sort_values(by=['ICUSTAY_ID', 'SEQ_NUM']).to_csv(
            os.path.join(dn, 'diagnoses.csv'), index=False)
    if verbose:
        sys.stdout.write('DONE!\n')


def break_up_procedures_by_subject(procedures, output_path, subjects=None, verbose=1):
    subjects = procedures.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for i, subject_id in enumerate(subjects):
        if verbose:
            sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i + 1, nb_subjects))
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        procedures.ix[procedures.SUBJECT_ID == subject_id].sort_values(by=['ICUSTAY_ID', 'SEQ_NUM']).to_csv(
            os.path.join(dn, 'procedures.csv'), index=False)
    if verbose:
        sys.stdout.write('DONE!\n')


# =======================================

def break_up_prescriptions_by_subject(prescriptions, output_path, subjects=None, verbose=1):
    subjects = prescriptions.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for i, subject_id in enumerate(subjects):
        if verbose:
            sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i + 1, nb_subjects))
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        prescriptions.ix[prescriptions.SUBJECT_ID == subject_id].sort_values(by='STARTDATE').to_csv(
            os.path.join(dn, 'prescriptions.csv'), index=False)

    if verbose:
        sys.stdout.write('DONE!\n')


# =======================================

def read_events_table_and_break_up_by_subject(mimic3_path, table, output_path, items_to_keep=None,
                                              subjects_to_keep=None, verbose=1):
    obs_header = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']
    if items_to_keep is not None:
        items_to_keep = set([str(s) for s in items_to_keep])
    if subjects_to_keep is not None:
        subjects_to_keep = set([str(s) for s in subjects_to_keep])

    class DataStats(object):
        def __init__(self):
            self.curr_subject_id = ''
            self.last_write_no = 0
            self.last_write_nb_rows = 0
            self.last_write_subject_id = ''
            self.curr_obs = []

    data_stats = DataStats()

    def write_current_observations():
        data_stats.last_write_no += 1
        data_stats.last_write_nb_rows = len(data_stats.curr_obs)
        data_stats.last_write_subject_id = data_stats.curr_subject_id
        dn = os.path.join(output_path, str(data_stats.curr_subject_id))
        try:
            os.makedirs(dn)
        except:
            pass
        fn = os.path.join(dn, 'events.csv')
        if not os.path.exists(fn) or not os.path.isfile(fn):
            f = open(fn, 'w')
            f.write(','.join(obs_header) + '\n')
            f.close()
        w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
        w.writerows(data_stats.curr_obs)
        data_stats.curr_obs = []

    for row, row_no, nb_rows in read_events_table_by_row(mimic3_path, table):
        if verbose and (row_no % 100000 == 0):
            if data_stats.last_write_no != '':
                sys.stdout.write('\rprocessing {0}: ROW {1} of {2}...last write '
                                 '({3}) {4} rows for subject {5}'.format(table, row_no, nb_rows,
                                                                         data_stats.last_write_no,
                                                                         data_stats.last_write_nb_rows,
                                                                         data_stats.last_write_subject_id))
            else:
                sys.stdout.write('\rprocessing {0}: ROW {1} of {2}...'.format(table, row_no, nb_rows))

        if (subjects_to_keep is not None and row['SUBJECT_ID'] not in subjects_to_keep):
            continue
        if (items_to_keep is not None and row['ITEMID'] not in items_to_keep):
            continue

        row_out = {'SUBJECT_ID': row['SUBJECT_ID'],
                   'HADM_ID': row['HADM_ID'],
                   'ICUSTAY_ID': '' if 'ICUSTAY_ID' not in row else row['ICUSTAY_ID'],
                   'CHARTTIME': row['CHARTTIME'],
                   'ITEMID': row['ITEMID'],
                   'VALUE': row['VALUE'],
                   'VALUEUOM': row['VALUEUOM']}
        if data_stats.curr_subject_id != '' and data_stats.curr_subject_id != row['SUBJECT_ID']:
            write_current_observations()
        data_stats.curr_obs.append(row_out)
        data_stats.curr_subject_id = row['SUBJECT_ID']

    if data_stats.curr_subject_id != '':
        write_current_observations()

    if verbose:
        sys.stdout.write('\rfinished processing {0}: ROW {1} of {2}...last write '
                         '({3}) {4} rows for subject {5}...DONE!\n'.format(table, row_no, nb_rows,
                                                                           data_stats.last_write_no,
                                                                           data_stats.last_write_nb_rows,
                                                                           data_stats.last_write_subject_id))
