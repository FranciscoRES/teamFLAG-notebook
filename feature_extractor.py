import pandas as pd
from sklearn.base import TransformerMixin

class FeatureExtractor(TransformerMixin):

    def __init__(self):
            pass

    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        X_df['yyyymm'] -= X_df['yyyymm'][0]
        X_df['yyyymm'] /= X_df['yyyymm'].iat[-1]
        X_df = X_df.replace(',','')
        return X_df.values