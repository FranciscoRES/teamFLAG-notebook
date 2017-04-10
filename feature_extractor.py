import pandas as pd


class FeatureExtractor(TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X_df, y):
        return self

	def transform(self, X_df):
		X_df['yyyymm'] -= X_df['yyyymm'][0]
		X_df['yyyymm'] /= X_df['yyyymm'].iget(-1)
	    return X_df.values