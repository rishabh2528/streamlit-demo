import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Step 1: Upload CSV
st.title("Fraud Detection System")
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Step 2: Display dropped columns and useful columns
    st.write("Data preview:")
    st.write(data.head())
    
    # Dropping unnecessary columns
    dropped_columns = ["Unnamed: 0", "trans_num", "street"]
    useful_columns = data.columns.difference(dropped_columns)
    st.write("Dropped Columns:", dropped_columns)
    st.write("Useful Columns:", useful_columns)

    # Preprocess data
    data.drop(columns=dropped_columns, inplace=True)
    df_processed = pd.get_dummies(data=data)

    # Step 3: Processing state
    st.write("Processing...")

    x_train = df_processed.drop(columns='is_fraud', axis=1)
    y_train = df_processed['is_fraud']
    
    # Logistic regression model
    lr = LogisticRegression(solver='liblinear', class_weight='balanced')
    lr.fit(x_train, y_train)

    # Step 4: Results (Confusion Matrix, Classification Report, Fraud Transactions)
    predictions = lr.predict(x_train)
    logicRegressionMatrix = confusion_matrix(y_train, predictions)
    
    st.write("Confusion Matrix")
    sns.heatmap(logicRegressionMatrix, annot=True, fmt='d', cmap='Oranges', xticklabels=['Non-Fraudulent', 'Fraudulent'], yticklabels=['Non-Fraudulent', 'Fraudulent'])
    st.pyplot(plt)
    
    # Fraud transactions
    fraud_transactions = data[data['is_fraud'] == 1]
    st.write("Fraud Transactions:")
    st.write(fraud_transactions)

    # Allow user to download the fraud transactions as CSV
    csv = fraud_transactions.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download fraud transactions as CSV", data=csv, file_name='fraud_transactions.csv', mime='text/csv')
