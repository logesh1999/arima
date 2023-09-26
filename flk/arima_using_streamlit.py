import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open('pickle/model.pkl', 'rb'))

def main():

        #df = pd.read_csv(w)
        start_date = st.date_input('start_date')
        end_date = st.date_input('end_date')
        pred = model.get_forecast('2020-12-01')
        pred_ci = pred.conf_int()
        p = pred.predicted_mean[start_date:end_date]
        # output = p[0]
        st.write(p)

if __name__ == '__main__':
    main()
