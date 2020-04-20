import csv
import os

import pandas as pd
from IPython.display import display
from dateutil.relativedelta import relativedelta


class Args(object):
    mimic3_path = ""
    output_path = ""


args = Args()
args.mimic3_path = "../../../mimic-iii-clinical-database-1.4"
args.output_path = "../../../mimic-iii-clinical-database-1.4/cut1000"


def read_csv_file(csv_file, root=args.mimic3_path, index_col=0):
    return pd.read_csv(os.path.join(root, csv_file), header=0, index_col=index_col)


def write_csv_file(df, csv_file, root=args.output_path, index=True, quoting=csv.QUOTE_MINIMAL):
    df.to_csv(os.path.join(root, csv_file), index=index, quoting=quoting)
    return


def read_and_cut(csv_file, col_to_cut, col_values, root=args.mimic3_path):
    iter_csv = pd.read_csv(os.path.join(root, csv_file), iterator=True, chunksize=100000, header=0,
                           index_col=0)
    df = pd.concat([chunk.loc[chunk[col_to_cut].isin(col_values)] for chunk in iter_csv])
    return df


def read_and_cut_multi_index(csv_file, col_values, root=args.mimic3_path):
    iter_csv = pd.read_csv(os.path.join(root, csv_file), iterator=True, chunksize=100000, header=0,
                           index_col=0)
    df = pd.concat([pd.merge(chunk, col_values, how='inner') for chunk in iter_csv])
    return df


def peek(csv_file, root=args.mimic3_path):
    df = pd.read_csv(os.path.join(root, csv_file), nrows=2)
    display(df.columns)


def get_age(start, end):
    diff = relativedelta(end, start)
    age = diff.years + diff.months / 12 + diff.days / 365 + diff.hours / (365 * 24) + diff.minutes / (
            365 * 24 * 60) + diff.seconds / (365 * 24 * 60 * 60)
    return age
