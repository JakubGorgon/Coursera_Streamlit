import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import streamlit as st
# df = pd.read_pickle("../../data/interim/mushrooms.pkl")

@st.cache_data(persist=True)
def split_data(df):
        X = df.drop(columns = ['poisonous'])
        y = df['poisonous']
        X_train, X_test, y_train, y_test = train_test_split(
            X,y, random_state=0, test_size=0.3)
        return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test = split_data(df)

