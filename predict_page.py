import streamlit as st
import pickle
import numpy as np
import pandas as pd


def load_model():
    with open('C:\electricity theft detection\SmartGridFraudDetection\saved_model.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
model = data["model"]

def show_predict_page():
    st.title("Electricity Theft Detection")
    st.write("""### We need some information to predict about customer""")

    st.write("""### Upload the csv file about customer information""")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
    # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Display the DataFrame
        st.write("Uploaded DataFrame:")
        st.write(df)

    ok=st.button("Predict")
    if ok:
        row=df.iloc[1]
        row_as_dataframe = pd.DataFrame(row).transpose()
        customer=         model.predict(row_as_dataframe)
        if(customer):
            st.write("Faithful Customer")
        else:
            st.write("Unfaithful Customer")