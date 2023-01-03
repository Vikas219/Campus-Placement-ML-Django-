import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

def read_input():
    placement = pd.read_csv('Placement.csv')
    placement_data = placement.copy()
    return placement, placement_data


def missing_values(placement_data):
    placement_data['salary'].fillna(value = 0,  inplace = True)
    placement_data.drop(['sl_no', 'ssc_b', 'hsc_b'], axis = 1, inplace = True)
    return placement_data


def outliers(placement_data):
    # take 75 %
    q1 = placement_data['hsc_p'].quantile(0.75)
    # take 25 %
    q2 = placement_data['hsc_p'].quantile(0.25)
    IQR = q1 - q2
    upper_limit = q1 + 1.5*IQR
    lower_limit = q2 - 1.5*IQR
    # filter the dataframe df_scores to retain records that lie in the permissible range.
    placement_data_filtered = placement_data[(placement_data['hsc_p'] >= lower_limit )  &  (placement_data['hsc_p'] <= upper_limit)]
    return placement_data_filtered


def preprocessing(placement_data_filtered):

    # creating a list of features that are to be label encoded
    label = ['gender', 'workex', 'specialisation', 'status']
    # creating instance of Label Encoder
    label_encoder = LabelEncoder()
    # Traverse the list
    for label_df in label:
        placement_data_filtered[label_df] = label_encoder.fit_transform(placement_data_filtered[label_df])

    dummy_hsc_s = pd.get_dummies(placement_data_filtered['hsc_s'],  prefix='dummy')
    dummy_degree_t = pd.get_dummies(placement_data_filtered['degree_t'],  prefix='dummy')
    placement_data_filtered_hsc_degree = pd.concat([placement_data_filtered, dummy_hsc_s, dummy_degree_t],  axis = 1)
    placement_data_filtered_hsc_degree.drop(['hsc_s', 'degree_t', 'salary'],  axis = 1,  inplace = True)

    x = placement_data_filtered_hsc_degree.drop(['status'], axis = 1)
    y = placement_data_filtered_hsc_degree.status

    return x, y


def print_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())

