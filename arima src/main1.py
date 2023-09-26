import pandas as pd
import pickle
import itertools
import numpy as np



model = pickle.load(open('model.pkl', 'rb'))

pred = model.get_forecast('2018-12-01')
pred_ci = pred.conf_int()
p = pred.predicted_mean['2017-01-01':'2017-12-01']
print(p)


