from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier




from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

def read_input():
    # print('1')
    placement = pd.read_csv('Placement.csv')
    # print('2')
    placement_data = placement.copy()
    # print('3')
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


def tell_me_status(dataframe):

    placement, placement_data = read_input()

    placement_data = pd.concat([placement_data, dataframe], ignore_index=True)
    # placement_data = placement_data.append(dataframe)

    placement_data = missing_values(placement_data)
    placement_data_filtered = outliers(placement_data)

    x, y = preprocessing(placement_data_filtered)

    # print(x.info())
    last_row = x.iloc[-1:]
    x = x.iloc[:-1]
    y = y.iloc[:-1]
    # print(last_row)
    # print(x)


    x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size = 0.8,  random_state = 42)

    ran_for_cl = RandomForestClassifier()
    ran_for_cl.fit(x_train,y_train)

    # y_pred  =  ran_for_cl.predict(x_test)

    y_pred = ran_for_cl.predict(last_row)

    print(y_pred)

    print('good')
    # print(y_pre)
    # print(dataframe.info())
    
    # y_predd = 0
    return y_pred



def predict(name, gender,ssc_p,ssc_b,hsc_p,hsc_b,hsc_s,degree_p,degree_t,workex,etest_p,specialisation,mba_p):

    data = [{
                'sl_no': 1000,
                'gender': gender,
                'ssc_p': ssc_p,
                'ssc_b': ssc_b,
                'hsc_p': hsc_p,
                'hsc_b': hsc_b,
                'hsc_s': hsc_s,
                'degree_p': degree_p,
                'degree_t': degree_t,
                'workex': workex,
                'etest_p': etest_p,
                'specialisation': specialisation,
                'mba_p': mba_p,
                'status': 'Placed',
                'salary': 68.0
    }]

    df = pd.DataFrame(data)
    ans = tell_me_status(df)
    return ans


# Create your views here.
def index(request):

    if request.method == 'POST':

        name = request.POST['name']
        gender = request.POST['gender']
        ssc_p = request.POST['ssc_p']
        ssc_b = request.POST['ssc_b']
        hsc_p = request.POST['hsc_p']
        hsc_b = request.POST['hsc_b']
        hsc_s = request.POST['hsc_s']
        degree_p = request.POST['degree_p']
        degree_t = request.POST.get('degree_t')
        workex = request.POST['workex']
        etest_p = request.POST['etest_p']
        specialisation = request.POST['specialisation']
        mba_p = request.POST['mba_p']

        ans = predict(name, gender,  float(ssc_p), ssc_b, float(hsc_p), hsc_b, hsc_s, float(degree_p), degree_t, workex, float(etest_p), specialisation, float(mba_p))

        if ans == 1:
            return render(request, 'placed.html')
        else:
            return render(request, 'unplaced.html')

    else:
        return render(request, 'index.html')


def about(request):
    return render(request, 'about.html')