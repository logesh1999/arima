from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)

'''
@app.route('/model/', methods=["GET"])
def predict():
    response = request.values.get()
    model = pickle.load(open('model1.pkl', 'rb'))

    if pd.to_datetime(response, format='%Y-%m-%d'):
        forecast = model.predict(response)
        return jsonify(forecast)
    else:
        return "Invalid date format"
        
    #return jsonify(forecast)
'''
@app.route('/', methods = ['GET'])
def model():
    #date = request.args.get('2020-01-02')
    #ch = pd.to_datetime(date, format='%Y-%m-%d')
    #data = datetime.strptime(date, format)
    model = pickle.load(open('../model1.pkl', 'rb'))
    test_date = '2020-01-02'
    if model:
        forecast = model.predict(test_date)
        return jsonify(forecast)
    else:
        return "Invalid date format"

if __name__ == '__main__':
    app.run(debug=True)