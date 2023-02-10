import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# HousePrice Prediction App
##### This app predicts the **House Prices**!
##### Data obtained from NTU CC0002.
""")

st.markdown(
    """
    <style>
    textarea {
        font-size: 3rem !important;
    }
    input {
        font-size: 3rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.sidebar.header('User Input Features')

# Collects user input features into dataframe
def user_input_features():
        bill_length_mm = st.sidebar.slider('Age (Years)', 0,100,50, step = 1)
        bill_depth_mm = st.sidebar.number_input('Area (Square Feet)', 500,2000,1250)
        flipper_length_mm = st.sidebar.slider('Quality', 0,10,5)
        data = {'Age': bill_length_mm,
                'Area': bill_depth_mm,
                'Quality': flipper_length_mm,
                }
        features = pd.DataFrame(data, index=[0])
        return features
input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('housing.csv')
penguins = penguins_raw.drop(columns=['Price'], axis=1)
df = pd.concat([input_df,penguins],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['Quality']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.write('### User Input features')

st.write(input_df)

# Reads in saved classification model
load_clf = pickle.load(open('module2.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(input_df)

st.subheader('Prediction')
st.subheader(prediction[0])