import pickle
#import pandas as pd
from datetime import datetime

date = input('Enter your date: ')
model = pickle.load(open('model1.pkl', 'rb'))
new_date = datetime.strptime(date, '%Y-%m-%d')

forecast = model.predict(new_date)
print(forecast)