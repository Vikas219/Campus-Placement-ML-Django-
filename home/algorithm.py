import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import algo_methods as am


def tell_me_status():

    placement, placement_data = am.read_input()

    placement_data = am.missing_values(placement_data)

    placement_data_filtered = am.outliers(placement_data)

    x, y = am.preprocessing(placement_data_filtered)

    x_train, x_test, y_train, y_test = train_test_split(x, y,  train_size = 0.8,  random_state = 42)

    ran_for_cl = RandomForestClassifier()

    ran_for_cl.fit(x_train,y_train)

    y_pred  =  ran_for_cl.predict(x_test)


    # dataframe = am.missing_values(dataframe)
    # dataframe = am.outliers(dataframe)
    # x_dataframe, y_dataframe = am.preprocessing(dataframe)
    # y_pred = ran_for_cl.predict(x_dataframe)
    
    print(y_pred)


tell_me_status()
