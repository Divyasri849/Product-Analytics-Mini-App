import streamlit as st
import pandas as pd
from data_ingestion import get_mongo_collection
from all_models import train_and_evaluate_model  
from pymongo import MongoClient

# ---------------------------------------------------------
# Load data from MongoDB
# ---------------------------------------------------------
def load_data():
    collection = get_mongo_collection()
    if collection is None:
        return pd.DataFrame()

    docs = list(collection.find({}, {"_id": 0}))
    return pd.DataFrame(docs)


# ---------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------
st.set_page_config(page_title="Product Analytics App", layout="wide")
st.title("Product Analytics Mini App")
st.write("API → MongoDB → Streamlit → ML")


# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
df = load_data()

if df.empty:
    st.error("No data found in MongoDB. Please run ingestion first.")
    st.stop()


# ---------------------------------------------------------
# 2. Exploratory Data Analysis (5.3)
# ---------------------------------------------------------
st.header("Exploratory Data Analysis")

st.subheader("Product Overview")
st.dataframe(df.head())

col1, col2 = st.columns(2)
# --- PRICE DISTRIBUTION  ---
with col1:
    st.subheader("Price Distribution (Histogram)")
    
    
    fig, ax = plt.subplots()
    
    ax.hist(df["price"], bins=15, edgecolor='black', color='#4CAF50')
    
    ax.set_xlabel("Price")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Product Prices")
    
    st.pyplot(fig)

# --- RATING DISTRIBUTION  ---
with col2:
    st.subheader("Distribution of Ratings")

    fig, ax = plt.subplots()

    ax.hist(df["rating"], bins=10, edgecolor='black')  

    ax.set_xlabel("Rating")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Product Ratings")

    st.pyplot(fig)


st.subheader("Products by Category")
category_count = df["category"].value_counts()
st.bar_chart(category_count)


# ---------------------------------------------------------
# 3. Model Training (Cached)
# ---------------------------------------------------------
@st.cache_resource
def cached_train_model():
    return train_and_evaluate_model()

st.header("Machine Learning Model")

if st.button("Retrain Model"):
    st.cache_resource.clear()
    st.success("Model retrained successfully!")

model, accuracy = cached_train_model()

if model is None:
    st.error("Model training failed. Check data.")
    st.stop()

st.write(f"**Model Accuracy:** {accuracy:.2f}")


# ---------------------------------------------------------
# 4. Prediction UI (5.5)
# ---------------------------------------------------------
st.header("Rating Prediction")

st.write("Enter product features:")

price = st.number_input("Price", min_value=0.0)
discount = st.number_input("Discount Percentage", min_value=0.0)
stock = st.number_input("Stock", min_value=0)
rating = None  # Not used for prediction in your model

if st.button("Predict Rating"):
    try:
        # Prepare input just like training features
        X_input = pd.DataFrame([{
            "price": price,
            "discountPercentage": discount,
            "stock": stock
        }])

        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0][1]

        label = "High Rated" if prediction == 1 else "Low Rated"

        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {probability:.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
