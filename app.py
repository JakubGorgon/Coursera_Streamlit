import streamlit as st
import pandas as pd; import numpy as np
import matplotlib.pyplot as plt; import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

def main():
    st.title("HELLO STREAMLIT")
    st.sidebar.title("Sidebar Check")
    st.markdown("Are your mushrooms poisonous? 🍄")
    
    
    
    
    

if __name__ ==  '__main__':
    main()