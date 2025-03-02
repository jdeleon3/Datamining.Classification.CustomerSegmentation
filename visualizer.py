import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

class Visualizer:
    """
    A class to visualize data
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def plot_histograms(self):
        figure = plt.figure(figsize=(20,20))
        self.df.hist(ax=figure.gca())        
        filename = './images/histograms.png'
        if(os.path.exists(filename)):
            os.remove(filename)    
        figure.savefig(filename)
        figure.clear()
        plt.close(figure)

    def plot_boxplots(self):
        for column in self.df.columns:
            box = sns.boxplot(x=self.df[column])
            filename = f'./images/boxplot_{column}.png'
            if(os.path.exists(filename)):
                os.remove(filename)
            box.figure.savefig(filename)
            box.figure.clear()
            plt.close(box.figure)
    
    def plot_decision_tree(self, model: DecisionTreeClassifier):
        plt.figure(figsize=(20,20))
        plot_tree(model, filled=True, feature_names=self.df.columns)
        filename = './images/decision_tree.png'
        if(os.path.exists(filename)):
            os.remove(filename)
        plt.savefig(filename)
        plt.close()


