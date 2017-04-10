from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
import feature_extractor
import regressor
import time

if __name__ == '__main__':
    tinit = time.time()
    print('Reading file ...')
    X_df = pd.read_csv('Data/BruteTrainData.csv')
#    target_cols = np.loadtxt(
#        'target.dta', dtype=bytes, delimiter=';').astype(str)
#        X_df = df.drop(target_cols, axis=1)
    y_array = pd.read_csv('Data/IndexTrain.csv', thousands = ',')['INDEX']
    #Watch out for ',' thousand separators!

    skf = ShuffleSplit(n_splits=2, test_size=0.5, random_state=67)
    print('Training file ...')

    fe = feature_extractor.FeatureExtractor()
    X_t = fe.transform(X_df)
    
    for train_is, test_is in skf.split(X_t, y_array):
        X_train_array = X_t[train_is]
        y_train_array = y_array[train_is]

        # Feature extraction
        fe = feature_extractor.FeatureExtractor()
        fe.fit(X_train_array, y_train_array)

        reg = regressor.Regressor()
        reg.fit(X_train_array, y_train_array)
        X_test_array = X_t[test_is]
        y_test_array = y_array[test_is]

        # Regression
        y_pred_array = reg.predict(X_test_array)
        print('rmse = ', np.sqrt(
            np.mean(np.square(y_test_array - y_pred_array))))
    tfin = time.time()
    print('time= ', tfin-tinit)