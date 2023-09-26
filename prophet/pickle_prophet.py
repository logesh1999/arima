from prophet import Prophet
import pickle
import pandas as pd

df = pd.read_csv('pro_2.csv')
df.columns=['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d')
df.set_index(['ds'])
model = Prophet(weekly_seasonality=True)
result = model.fit(df)
#print(result)
f = pickle.dump(result,open('model1.pkl', 'wb'))