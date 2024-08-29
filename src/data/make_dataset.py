from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st

  

# fetch dataset 
def load_and_encode_data():
    mushroom = fetch_ucirepo(id=73) 
    
    # data (as pandas dataframes) 
    X = mushroom.data.features 
    y = mushroom.data.targets 

    df = pd.concat([X, y], axis = 1)

    encoder = LabelEncoder()

    cols = df.columns

    for col in cols:
        df[col] = encoder.fit_transform(df[col])

    df.to_pickle("../../data/interim/mushrooms.pkl")

