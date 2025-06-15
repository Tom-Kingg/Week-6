import streamlit as st
import os
import pandas as pd
import numpy as np
import zipfile
import pickle
from PIL import Image

from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(page_title="Timelytics", layout="wide")

st.title("Timelytics: Predict Order-to-Delivery Time")
st.caption("Forecast delivery days using real e-commerce data and machine learning ensemble techniques (XGBoost + Random Forest + SVR).")

# Set Kaggle API credentials (use secrets in deployment)
os.environ['KAGGLE_USERNAME'] = "your_username"  # 🔁 Replace with your Kaggle username
os.environ['KAGGLE_KEY'] = "your_key"            # 🔁 Replace with your Kaggle API key

# Download and extract Kaggle dataset (only once)
@st.cache_resource
def download_and_extract_data():
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)
        os.system("kaggle datasets download -d olistbr/brazilian-ecommerce -p data")
        with zipfile.ZipFile("data/brazilian-ecommerce.zip", 'r') as zip_ref:
            zip_ref.extractall("data")
    return True

# Load + preprocess + train + save model
@st.cache_resource
def prepare_model():
    download_and_extract_data()
    orders = pd.read_csv("data/olist_orders_dataset.csv")
    items = pd.read_csv("data/olist_order_items_dataset.csv")
    cust = pd.read_csv("data/olist_customers_dataset.csv")

    df = orders.merge(items, on="order_id").merge(cust, on="customer_id")
    df = df[["order_approved_at", "order_delivered_customer_date", "price", "freight_value", "customer_state"]].dropna()

    df["order_approved_at"] = pd.to_datetime(df["order_approved_at"])
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])
    df["delivery_days"] = (df["order_delivered_customer_date"] - df["order_approved_at"]).dt.days
    df = df[df["delivery_days"] > 0]

    df["month"] = df["order_approved_at"].dt.month
    df["dow"] = df["order_approved_at"].dt.dayofweek
    df["year"] = df["order_approved_at"].dt.year
    df["customer_state"] = df["customer_state"].astype("category").cat.codes

    X = df[["dow", "month", "year", "price", "freight_value", "customer_state"]]
    y = df["delivery_days"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model1 = RandomForestRegressor()
    model2 = XGBRegressor()
    model3 = SVR()
    voting = VotingRegressor([('rf', model1), ('xgb', model2), ('svr', model3)])
    voting.fit(x_train, y_train)

    with open("voting_model.pkl", "wb") as f:
        pickle.dump(voting, f)
    
    return voting

# Load model
voting_model = prepare_model()

# Sidebar input
with st.sidebar:
    st.header("📥 Input Parameters")
    purchase_dow = st.number_input("Purchased Day of Week (0=Mon)", min_value=0, max_value=6, value=3)
    purchase_month = st.number_input("Purchased Month", min_value=1, max_value=12, value=1)
    year = st.number_input("Purchase Year", value=2018)
    price = st.number_input("Price (BRL)", value=150.00)
    freight_value = st.number_input("Freight Value (BRL)", value=25.00)
    customer_state = st.number_input("Customer State Code (0–26)", value=10)
    submit = st.button("Predict Delivery Time")

# Prediction function
def predict_delivery(dow, month, year, price, freight, state):
    features = np.array([[dow, month, year, price, freight, state]])
    prediction = voting_model.predict(features)
    return round(prediction[0])

# Show prediction
if submit:
    with st.spinner("Predicting..."):
        result = predict_delivery(purchase_dow, purchase_month, year, price, freight_value, customer_state)
        st.success(f"📦 Estimated Delivery Time: **{result} days**")

# Sample data
st.header("📊 Sample Dataset Preview")
sample_data = pd.DataFrame({
    "Purchased Day": [2, 1, 3],
    "Month": [4, 6, 8],
    "Year": [2018, 2018, 2017],
    "Price": [150.0, 89.5, 230.0],
    "Freight": [25.0, 12.0, 40.0],
    "Customer State Code": [10, 15, 8],
})
st.write(sample_data)
