from flask import Flask, request, jsonify
import pandas as pd
import pickle
from prophet import Prophet


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def prediction():

    if request.method == 'GET':
        date = request.values.get('Date')
    if request.method == 'POST':
        model = pickle.load(open('model1.pkl', 'rb'))
        h = pd.to_datetime(date, format="%Y-%m-%d")
        forecast = model.predict(h)
        return forecast


if __name__ == '__main__':
    app.run(debug = True)