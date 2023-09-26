from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/prediction', methods = ['POST'])
def predict():

    if request.method == "POST":
        request_data = request.get_json()
        date = request_data['Date']
        #test = '2020-01-02'
        #print(test)
        h = pd.to_datetime(date, format='%Y-%m-%d')
        model = pickle.load(open('model1.pkl', 'rb'))
        forecast = model.predict(h)
        return forecast
    else:
         return "Invalid date format"

if __name__ == '__main__':
    app.run(debug = True)