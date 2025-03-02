import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

class Model:
    """
    A class to create a model
    """

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        self.train_df = train_df
        self.test_df = test_df
        self.dtc_model = DecisionTreeClassifier(max_depth=3, max_features=3)
        self.rfc_model = RandomForestClassifier(n_estimators=100, max_depth=5)
        self.target = 'Segmentation'

    def train_decisiontree_model(self):
        print(self.train_df)
        X_train = self.train_df.drop([self.target], axis=1)
        y_train = self.train_df[self.target]
        #X_train = self.train_df.drop([self.target], axis=1)
        #y_train = self.train_df[self.target]
        #print(X_train)
        #print(y_train)
        self.dtc_model.fit(X_train, y_train)

    def predict_decisiontree(self):
        X_test = self.test_df.drop([self.target], axis=1)
        y_test = self.test_df[self.target]
        
        y_predicted = self.dtc_model.predict(X_test)
        print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_predicted)}")
        print(f"Classification Report: \n{classification_report(y_test, y_predicted)}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_predicted)}")

    def evaluate_decisiontree(self):
        X_test = self.test_df.drop([self.target], axis=1)
        y_test = self.test_df[self.target]
        return self.dtc_model.score(X_test, y_test)    
    
    def train_randomforest_model(self):
        X_train = self.train_df.drop([self.target], axis=1)
        y_train = self.train_df[self.target]
        self.rfc_model.fit(X_train, y_train)

    def predict_randomforest(self):
        X_test = self.test_df.drop([self.target], axis=1)
        y_test = self.test_df[self.target]
        
        y_predicted = self.rfc_model.predict(X_test)
        print(f"Random Forest Accuracy: {accuracy_score(y_test, y_predicted)}")
        print(f"Classification Report: \n{classification_report(y_test, y_predicted)}")
        print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_predicted)}")