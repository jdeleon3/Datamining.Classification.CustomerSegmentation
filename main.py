import pandas as pd
import numpy as np
from datahandler import DataHandler
from model import Model
from visualizer import Visualizer

dh = DataHandler()

dh.clean_data()
dh.balance_data()
dh.encode_data()
#dh.inspect_train_data()
v = Visualizer(dh.get_train_data())


m = Model(dh.get_train_data(), dh.get_test_data())
m.train_decisiontree_model()
m.predict_decisiontree()
m.evaluate_decisiontree()

m.train_randomforest_model()
m.predict_randomforest()

v.plot_decision_tree(m.dtc_model)
