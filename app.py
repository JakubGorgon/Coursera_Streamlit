import streamlit as st
import pandas as pd; import numpy as np
import matplotlib.pyplot as plt; import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import time

from src.models.build_models import split_data

def main():
    st.title("Mushroom Classification App")
    st.sidebar.subheader("Choose Classification Method")
    st.markdown("Are your mushrooms poisonous? 🍄")
    
    classifiers = ("Select and option", "SVM", "Logistic Regression", "Random Forest")
    classifier = st.sidebar.selectbox("Classifier", classifiers, index=0)
    
    # metrics_list = ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve")
    # metric = st.sidebar.multiselect("Metrics to Plot", metrics_list)
    
    @st.cache_data(persist=True)
    def load_data():
        df = pd.read_pickle("data/interim/mushrooms.pkl")
        return df
    
    df = load_data()
    
    X_train, X_test, y_train, y_test = split_data(df)
    
    def plot_metrics(metrics_list):
                
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, yhat)
            disp = ConfusionMatrixDisplay(cm, display_labels=['edible', 'poisonous'])
            disp.plot(ax=ax)
            st.pyplot(fig)

    
        if "ROC Curve" in metrics_list:
            st.subheader("ROC curve")
            fig, ax = plt.subplots()
            disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
            disp.plot(ax=ax)
            st.pyplot(fig)
            
        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            disp = PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
            disp.plot(ax=ax)
            st.pyplot(fig)
            
            
    if classifier == "SVM":
        st.sidebar.subheader("Model Hyperparameters")
        c = st.sidebar.number_input("Regularization Parameter", 0.01, 20.0, 
                                    step=0.01, key="c", value=10.0)
        k = st.sidebar.radio("Kernel", ("rbf", "linear"), key= 'k')
        g = st.sidebar.radio("Gamma (Kernel Coefficient)", 
                             ("scale", "auto"))
        
        metrics_list = ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve")
        metric = st.sidebar.multiselect("Metrics to Plot", metrics_list, default="Confusion Matrix")

        if st.sidebar.button("Classify", key = 'Classify'):
            st.subheader("SVM Classification Results")
            SVM = SVC(C=c, kernel=k, gamma=g)
            model = SVM.fit(X_train, y_train)
            yhat = model.predict(X_test)
            
            acc = accuracy_score(y_test, yhat)
            st.write(f"Accuracy: {round(acc,2)}")
            
            recall = recall_score(y_test, yhat)
            st.write(f"Recall: {round(recall,2)}")
            
            precision = precision_score(y_test,yhat)
            st.write(f"Precision: {round(precision,2)}")
            
            plot_metrics(metric)

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        c = st.sidebar.number_input("Regularization Parameter", 0.00, 20.0, 
                                    step=0.01, key="c_lr", value=1.0)
        i = st.sidebar.slider("Maximum Number of Iterations", 10, 1000, value=100, key='i')
        p = st.sidebar.radio("Penalty", ("l1", "l2", "elasticnet", None), key= 'p', index = 1)
        s = st.sidebar.radio("Solver", ("lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"), index = 0)
        if p == "elasticnet":
            l1 = st.sidebar.number_input("L1 ratio", 0.01, 0.99, step = 0.01, value=0.5)
        metrics_list = ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve")
        metric = st.sidebar.multiselect("Metrics to Plot", metrics_list, default="Confusion Matrix")

        if st.sidebar.button("Classify", key = 'Classify'):
            st.subheader("Logistic Regression Classification Results")
            
            start = time.time()
            
            try:
                if p == 'elasticnet':
                    Logreg = LogisticRegression(C=c, penalty=p, solver=s,
                                                max_iter = i, l1_ratio=l1)
                else:
                    Logreg = LogisticRegression(C=c, penalty=p, solver=s,
                                                max_iter=i)
                model = Logreg.fit(X_train, y_train)
                
                end = time.time()
                
                yhat = model.predict(X_test)
                
                acc = accuracy_score(y_test, yhat)
                st.write(f"Accuracy: **{round(acc, 2)}**")
                
                recall = recall_score(y_test, yhat)
                st.write(f"Recall: {round(recall,2)}")
                
                precision = precision_score(y_test,yhat)
                st.write(f"Precision: {round(precision,2)}")
                
                plot_metrics(metric)
                
                st.write(f"It took {round((end-start)*1000)} ms to train the classifier.")
                
            except ValueError as e:
                st.error(f"Oops, looks like {e}")
 
    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        t = st.sidebar.number_input("Number of Trees in the Forest",1, 5000,step=1  ,key="t", value=100)
        m = st.sidebar.number_input("Max Depth of a Tree", 1, 20, value=None, placeholder="Default (infinite)")
        c = st.sidebar.radio("Criterion", ("gini", "entropy", "log_loss"), key= 'cr')
        
        metrics_list = ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve")
        metric = st.sidebar.multiselect("Metrics to Plot", metrics_list, default="Confusion Matrix")

        if st.sidebar.button("Classify", key = 'Classify'):
            start = time.time()

            st.subheader("Random Forest Classification Results")
            RF = RandomForestClassifier(n_estimators=t, criterion=c, max_depth=m)
            model = RF.fit(X_train, y_train)
            
            end = time.time()

            yhat = model.predict(X_test)
            
            acc = accuracy_score(y_test, yhat)
            st.write(f"Accuracy: {round(acc,2)}")
            
            recall = recall_score(y_test, yhat)
            st.write(f"Recall: {round(recall,2)}")
            
            precision = precision_score(y_test,yhat)
            st.write(f"Precision: {round(precision,2)}")
            
            plot_metrics(metric)
            st.write(f"It took {round((end-start)*1000)} ms to train the classifier.")

    
    
    if st.sidebar.checkbox("Show Raw Data"):
        df
        
if __name__ ==  '__main__':
    main()