import pandas as pd
import warnings
import itertools
import numpy as np
import statsmodels.api as sm
import pickle

df = pd.read_csv('order.csv')
print(df.describe())
#print(df.head())
#print(df)

l = list(df['Sales'])
f = l[24:36]
print(f)

df['Order Date'] = pd.to_datetime(df['Order Date'], format='%Y-%m-%d')
df.set_index(['Order Date'], inplace=True)
'''
import matplotlib.pyplot as plt

df.plot()
plt.ylabel('Sales')
plt.xlabel('Date')
plt.show()
'''
q = d = range(0, 2)
p = range(0, 4)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

train_data = df['2014-01-01':'2015-12-01']
test_data = df['2016-01-01':'2017-12-01']

warnings.filterwarnings("ignore")

AIC = []
SARIMAX_model = []
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(train_data,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            #print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
            AIC.append(results.aic)
            SARIMAX_model.append([param, param_seasonal])
        except:
            continue


print('The smallest AIC is {} for model SARIMAX{}x{}'.format(min(AIC), SARIMAX_model[AIC.index(min(AIC))][0],SARIMAX_model[AIC.index(min(AIC))][1]))

mod = sm.tsa.statespace.SARIMAX(train_data,
                                order=SARIMAX_model[AIC.index(min(AIC))][0],
                                seasonal_order=SARIMAX_model[AIC.index(min(AIC))][1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

#pickle.dump(results, open('model.pkl','wb'))

pred0 = results.get_prediction(start='2015-01-01', dynamic=False)
pred0_ci = pred0.conf_int()
print(pred0_ci)

pred1 = results.get_prediction(start='2015-01-01', dynamic=True)
pred1_ci = pred1.conf_int()
print(pred1_ci)
print("==================================")

pred2 = results.get_forecast('2018-12-01')
pred2_ci = pred2.conf_int()
p = pred2.predicted_mean['2017-01-01':'2017-12-01']
print(p)
'''
pred3 = results.get_forecast('2018-12-01').summary_frame()
print(pred3)

s = pd.DataFrame({'predicted':p, 'Actual':f})
print(s)

model = pickle.load(open('model.pkl', 'rb'))

pred4 = model.get_forecast('2018-12-01')
'''