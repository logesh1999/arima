import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('order.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
grpBy = df.groupby(['Product_Code']).size().reset_index(name='counts').sort_values(['counts'],ascending=False)
#print(grpBy.head(10))
prod1359DF = df.loc[df['Product_Code'] == 'Product_1359'].sort_values(['Date'],ascending=False)
prod1359DF = prod1359DF.drop(columns=['Warehouse','Product_Code','Product_Category'])
prod1359DF.index=pd.to_datetime(prod1359DF.Date,format='%Y/%m/%d')
prod1359DF.drop(columns=['Date'],inplace=True)
prod1359DF['Order_Demand'] = prod1359DF['Order_Demand'].astype(str)
prod1359DF['Order_Demand'] = prod1359DF['Order_Demand'].map(lambda x: x.lstrip('(').rstrip(')'))
prod1359DF['Order_Demand'] = prod1359DF['Order_Demand'].astype(int)
#print(prod1359DF)
prod1359DmndMnth = prod1359DF.resample('M').sum()
print(prod1359DmndMnth.tail(12))
prod1359DmndMnth.drop(prod1359DmndMnth.loc[prod1359DmndMnth['Order_Demand']==100000].index,inplace=True)
prod1359DmndMnth.Order_Demand.plot(figsize=(13,6), title= 'Product 1359 Demand', fontsize=14,color="Green")
#plt.show()
prod1359Train = prod1359DmndMnth[:'2016-03-31']
prod1359Test = prod1359DmndMnth['2016-04-30':]
prod1359Train.Order_Demand.plot(figsize=(13,6), title= 'Product 1359 - Train and Test', fontsize=12,color="Green")
prod1359Test.Order_Demand.plot(figsize=(13,6), title= 'Product 1359 - Train and Test', fontsize=12)
plt.show()

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
y_hat_avg = prod1359Test.copy()
fit2 = SimpleExpSmoothing(np.asarray(prod1359Train['Order_Demand'])).fit(smoothing_level=0.6,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(prod1359Test))
plt.figure(figsize=(14,6))
plt.plot(prod1359Train['Order_Demand'], label='Train',color="Green")
plt.plot(prod1359Test['Order_Demand'], label='Test')
plt.plot(y_hat_avg['SES'], label='SES',color="Red")
plt.title("Simple Smoothing")
plt.legend(loc='best')
plt.show()

import math
from sklearn.metrics import mean_squared_error
smooth_rms = math.sqrt(mean_squared_error(prod1359Test.Order_Demand, y_hat_avg.SES))
print(smooth_rms)

exp_hat_avg = prod1359Test.copy()
fit1 = ExponentialSmoothing(np.asarray(prod1359Train['Order_Demand']) ,seasonal_periods=4 ,trend='additive', seasonal='additive',).fit()
exp_hat_avg['Exp_Smooth'] = fit1.forecast(len(prod1359Test))
plt.figure(figsize=(14,6))
plt.plot( prod1359Train['Order_Demand'], label='Train',color="Green")
plt.plot(prod1359Test['Order_Demand'], label='Test')
plt.plot(exp_hat_avg['Exp_Smooth'], label='Exp_Smooth',color="Red")
plt.legend(loc='best')
plt.title("Exponential Smoothing");
plt.show()

import math
from sklearn.metrics import mean_squared_error
exp_rms = math.sqrt(mean_squared_error(prod1359Test.Order_Demand, exp_hat_avg.Exp_Smooth))
print(exp_rms)

import statsmodels.api as sm
sm.tsa.seasonal_decompose(prod1359Train.Order_Demand).plot()
result = sm.tsa.stattools.adfuller(prod1359Train.Order_Demand)
plt.show()

decom_hat_avg = prod1359Test.copy()

fit1 = Holt(np.asarray(prod1359Train['Order_Demand'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
decom_hat_avg['Decom'] = fit1.forecast(len(prod1359Test))

plt.figure(figsize=(14,6))
plt.plot(prod1359Train['Order_Demand'], label='Train',color="Green")
plt.plot(prod1359Test['Order_Demand'], label='Test')
plt.plot(decom_hat_avg['Decom'], label='Decom',color="Red")
plt.legend(loc='best')
plt.title("Decomposition");
plt.show()

import math
from sklearn.metrics import mean_squared_error

decom_rms = math.sqrt(mean_squared_error(prod1359Test.Order_Demand, decom_hat_avg.Decom))
print(decom_rms)

n_groups = 3
test_mse = (smooth_rms, exp_rms, decom_rms)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects2 = plt.bar(index + bar_width, test_mse, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Test')

plt.ylabel('Root Mean Square Error')
plt.title('Root Mean Square Error for Product 1359')
plt.xticks(index + bar_width, ('Simple Smoothing', 'Exponential Smoothing', 'Decomposition'))
plt.show()

