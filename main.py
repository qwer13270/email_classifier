# project: p7
# submitter: chen925@wisc.edu
# partner: none
# hours: 20
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
class UserPredictor:
    def fit(self, train_users, train_logs, train_y):
        df = combine_data(train_users, train_logs)
        train_y = train_y.set_index("user_id")
        df["result"] = train_y["y"]
        self.model = Pipeline([
            ("poly", PolynomialFeatures(degree = 3)),
            ("std", StandardScaler()),
            ("lr", LogisticRegression(fit_intercept=False)),
        ])
        self.x_col = ["past_purchase_amt", "seconds", "age"]
        self.y_col = "result"
        train, test = train_test_split(df)
        self.model.fit(train[self.x_col], train[self.y_col])
    def predict(self, test_users, test_logs):
        df = combine_data(test_users, test_logs)
        df["prediction"] = self.model.predict(df[self.x_col])
        return df["prediction"].values
    
def combine_data(train_users, train_logs):
    train_logs = train_logs.set_index("user_id")
    train_users = train_users.set_index("user_id")
    p7_test = train_logs.groupby("user_id").mean()
    train_users["seconds"] = p7_test["seconds"]
    train_users["seconds"] = train_users["seconds"].replace(np.nan, 0)
    return train_users