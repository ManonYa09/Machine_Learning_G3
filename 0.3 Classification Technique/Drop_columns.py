from sklearn.base import BaseEstimator, TransformerMixin

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()  # Avoid modifying the original data
        if self.columns_to_drop:
            X = X.drop(columns=self.columns_to_drop, errors="ignore")
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return [col for col in input_features if col not in self.columns_to_drop]