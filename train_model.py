import sqlite3
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

def load_dataset():
    conn = sqlite3.connect('customer_churn.db')
    customer = pd.read_sql("SELECT * FROM db_customer", conn)
    subscription = pd.read_sql("SELECT * FROM db_subscription", conn)
    support = pd.read_sql("SELECT * FROM db_support", conn)
    conn.close()

    # Preprocessing
    customer = customer.rename(columns={"name": "customer_name"})
    customer["gender"] = customer["gender"].replace({"Men": "Male", "Women": "Female"})

    state_country = customer.dropna(subset=["country"]).drop_duplicates("state").set_index("state")["country"]
    customer["country"] = customer["country"].fillna(customer["state"].map(state_country))

    date_cols = ["subscription_start_date", "renewal_date", "cancellation_date"]
    for col in date_cols:
        subscription[col] = pd.to_datetime(subscription[col], errors="coerce")

    subscription["churn_flag"] = np.where(subscription["cancellation_date"].notna(), 1, 0)

    support["complaint_date"] = pd.to_datetime(support["complaint_date"], errors="coerce")
    support["complaint_count"] = support.groupby("customerid")["customerid"].transform("count")
    support_summary = support.sort_values("complaint_date").drop_duplicates("customerid", keep="last")

    df = subscription.merge(customer, on="customerid", how="left").merge(support_summary, on="customerid", how="left")

    today = pd.Timestamp.today()
    df["tenure_days"] = np.where(
        df["cancellation_date"].notna(),
        (df["cancellation_date"] - df["subscription_start_date"]).dt.days,
        (today - df["subscription_start_date"]).dt.days
    )
    df["tenure_days"] = df["tenure_days"].fillna(0).clip(lower=1)
    df["complaint_count"] = df["complaint_count"].fillna(0)
    df["csat_score"] = df["csat_score"].fillna(70) # default neutral CSAT
    df["escalations"] = df["escalations"].fillna("N")

    return df

def train_and_save():
    df = load_dataset()

    feature_cols = [
        "plan_type", "contract_type", "subscription_type", "gender", "country",
        "monthly_charges", "cltv", "tenure_days", "complaint_count", "csat_score", "escalations"
    ]
    categorical_cols = ["plan_type", "contract_type", "subscription_type", "gender", "country", "escalations"]
    numerical_cols = ["monthly_charges", "cltv", "tenure_days", "complaint_count", "csat_score"]

    X = df[feature_cols]
    y = df["churn_flag"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Model Training Complete!")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test ROC-AUC: {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Feature Importance extraction
    ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numerical_cols + cat_feature_names

    importances = pipeline.named_steps['classifier'].feature_importances_
    fi_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances}).sort_values('importance', ascending=False)
    print("\nTop 10 Feature Importances:\n", fi_df.head(10))

    # Save model and metadata
    model_data = {
        'pipeline': pipeline,
        'feature_cols': feature_cols,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'feature_importances': fi_df
    }

    with open('churn_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("\nSaved churn_model.pkl successfully!")

if __name__ == "__main__":
    train_and_save()
