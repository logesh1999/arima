from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/' , methods = ['GET', "POST"])
def main():

    #forecast = request.args.get('forecast')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    pred = model.get_forecast('2020-12-01')
    pred_ci = pred.conf_int()
    p = pred.predicted_mean[start_date:end_date]
    # output = p[0]
    return str(p)




if __name__ == "__main__":
    app.run(debug = True)

