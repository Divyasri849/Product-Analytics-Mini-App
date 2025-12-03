import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_ingestion import get_mongo_collection

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False


def train_and_evaluate_model():
    """
    Trains multiple ML models on MongoDB product data:
        - Logistic Regression
        - Decision Tree
        - Random Forest
        - XGBoost (optional)
    Returns the best model and its accuracy.
    """

    # ----------------------------------------------
    # 1.Loading data from MongoDB
    # ----------------------------------------------
    collection = get_mongo_collection()
    if collection is None:
        print("Failed to connect to MongoDB for training.")
        return None, None
    
    data = list(collection.find())
    if not data:
        print("No data found for training.")
        return None, None
        
    df = pd.DataFrame(data)

    # ----------------------------------------------
    # 2.Data Cleaning & Preparation
    # ----------------------------------------------
    #
    features = ['price', 'discountPercentage', 'stock'] 

    df[features] = df[features].apply(pd.to_numeric, errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=features + ['rating'])

    if df.empty:
        print("Data is empty after cleaning. Cannot train model.")
        return None, None

    # Target variable: High rating if >= 4.0
    df['is_high_rated'] = (df['rating'] >= 4.0).astype(int)
    
    X = df[features]
    y = df['is_high_rated']

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ----------------------------------------------
    # 3. Define All Models
    # ----------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )

    # ----------------------------------------------
    # 4. Train + Evaluate All Models
    # ----------------------------------------------
    accuracies = {}
    best_model = None
    best_accuracy = -1
    best_name = None

    print("\n---- MODEL ACCURACY RESULTS ----")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc

        print(f"{name}: {acc:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_name = name

    print("---------------------------------")
    print(f"BEST MODEL: {best_name}  (Accuracy = {best_accuracy:.4f})")
    print("---------------------------------")

    # Attach features list to the best model for Streamlit
    best_model.features = features
      
    # ------------------------------------------------------------
    print("---- TEST DATA (X_test) ----")
    print(X_test.head(20))

    # ------------------------------------------------------------
    # PRINT PREDICTION VS ACTUAL
    # ------------------------------------------------------------
    print("\n---- PREDICTION vs ACTUAL ----")

    y_pred_best = best_model.predict(X_test)

    comparison_df = X_test.copy()
    comparison_df["Actual"] = y_test.values
    comparison_df["Predicted"] = y_pred_best
    comparison_df["Correct"] = comparison_df["Actual"] == comparison_df["Predicted"]

    print(comparison_df.head(20))


    return best_model, best_accuracy


if __name__ == "__main__":
    train_and_evaluate_model()
