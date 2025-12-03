# Product-Analytics-Mini-App
End-to-end Product Analytics Mini-App with data ingestion, MongoDB storage, EDA, and multiple machine learning models for rating prediction.Includes Streamlit UI for visual analysis and real-time predictions
# Prediction Problem
 . The goal is to classify whether a product is high-rated.
 . A product is labeled 1 (High Rated) if its rating is ≥ 4.0, otherwise 0 (Low Rated).
# Features Used
 . The model uses the following numerical features:
 . price
 . discountPercentage
 . stock
# Model Chosen and Why
 . Among Logistic Regression, Decision Tree, Random Forest, and XGBoost:
 . Logistic Regression achieved the highest accuracy.
This is because:
 . It performs well on small datasets
 . It works best with numerical features
 . It avoids overfitting unlike tree models on limited data
# Trade-Offs / Limitations
 . The dataset contains only 194 cleaned samples, which limits learnability.
 . Complex models (Random Forest, XGBoost) cannot perform well without more data.
 . Accuracy cannot significantly improve unless additional features or more data points are added.
 . The model does not use text features (e.g., product description), category encodings, or embeddings that might improve prediction power.
# Project Overview
  - Fetches product data from dummyjson API
  - Stores in MongoDB with upsert
  - EDA via Streamlit
  - ML models: Logistic Regression, Decision Tree, Random Forest, XGBoost
  - Best model auto selected
  - Prediction verification
# Project Structure
 - project/
 - data_fetching_and_ingestion.py
 - model_building.py
 - streamlit_app.py
 - requirements.txt
 - README.md
# Setup Instructions
 Step 1 — Install Python 3.11.9
 Step 2 — Create virtual environment
 Step 3 — Install dependencies (pip install -r requirements.txt)
 Step 4 — Start MongoDB
 Step 5 — Run ingestion(python data_fetching_and_ingestion.py)
 Step 6 — Train ML models(python model_building.py)
 Step 7 — Run Streamlit(streamlit run streamlit_app.py)
# EDA Features
 - Product table
 - Price histogram
 - Rating histogram
 - Category bar count
# Machine Learning Details
 Target: High Rated (rating >= 4 → 1)
 Features: price, discountPercentage, stock
 Models compared: Logistic, Decision Tree, Random Forest, XGBoost
# Prediction Validation
 Test dataset printed
 Actual vs Predicted printed
 Correct/Incorrect included
# Assumptions & Tradeoffs
- API schema stable
- MongoDB local
- Only numeric features used
# Requirements
 pandas
 pymongo
 requests
 scikit-learn
 xgboost
 streamlit
 matplotlib

