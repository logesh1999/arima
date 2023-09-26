import streamlit as st
#import pickle
from flask import Flask
from prophet import Prophet
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

app = Flask(__name__)

@app.route('/predict')
def main():
    w = st.file_uploader("Upload a CSV file", type="csv")
    if w:

        df = pd.read_csv(w, parse_dates=True)
        df.columns = ['ds', 'y']
        #m = st.text_input('Please enter the forecast date')
        st.write("The DATAFRAME YOU UPLOADED",df)
        df['ds'] = pd.to_datetime(df['ds'])
        df.set_index(['ds'])
        model = Prophet(weekly_seasonality=True)
        model.fit(df)
        future = model.make_future_dataframe(periods=12, freq='m')
        #future = model.make_future_dataframe(periods=365)

        #st.write(future)
        forecast = model.predict(future)
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

        #st.pyplot(model.plot_components(forecast))

        metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()

        metric_df.dropna(inplace=True)
        st.write(metric_df)
        st.write(r2_score(metric_df.y, metric_df.yhat))
        st.write(mean_squared_error(metric_df.y, metric_df.yhat))
        st.write(mean_absolute_error(metric_df.y, metric_df.yhat))
        st.pyplot(model.plot(forecast))


if __name__ == '__main__':
    main()