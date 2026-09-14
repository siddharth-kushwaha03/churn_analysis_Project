
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3

# Page configuration
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Analysis Dashboard")
st.write("Interactive dashboard based on Customer Churn Analysis")

# Load and clean data
@st.cache_data
def load_data():
    conn = sqlite3.connect("customer_churn.db")

    customer = pd.read_sql("SELECT * FROM db_customer", conn)
    subscription = pd.read_sql("SELECT * FROM db_subscription", conn)
    support = pd.read_sql("SELECT * FROM db_support", conn)

    conn.close()

    # Customer cleaning
    customer = customer.rename(columns={"name": "customer_name"})
    customer = customer.drop(columns=["interests", "pincode"], errors="ignore")
    customer["dob"] = pd.to_datetime(customer["dob"], errors="coerce")
    customer["gender"] = customer["gender"].replace({
        "Men": "Male",
        "Women": "Female"
    })

    state_country = (
        customer.dropna(subset=["country"])
        .drop_duplicates("state")
        .set_index("state")["country"]
    )

    customer["country"] = customer["country"].fillna(
        customer["state"].map(state_country)
    )

    # Subscription cleaning
    date_cols = [
        "subscription_start_date",
        "renewal_date",
        "cancellation_date"
    ]

    for col in date_cols:
        subscription[col] = pd.to_datetime(
            subscription[col], errors="coerce"
        )

    subscription["churn_flag"] = np.where(
        subscription["cancellation_date"].notna(), 1, 0
    )

    # Support cleaning
    support["complaint_date"] = pd.to_datetime(
        support["complaint_date"], errors="coerce"
    )

    support = support.drop(
        columns=["col_1", "comment"], errors="ignore"
    )

    support["complaint_count"] = (
        support.groupby("customerid")["customerid"].transform("count")
    )

    support = (
        support.sort_values("complaint_date")
        .drop_duplicates("customerid", keep="last")
    )

    # Merge tables
    df = (
        subscription
        .merge(customer, on="customerid", how="left")
        .merge(support, on="customerid", how="left")
    )

    # Feature engineering
    today = pd.Timestamp.today()

    df["tenure_days"] = np.where(
        df["cancellation_date"].notna(),
        (df["cancellation_date"] -
         df["subscription_start_date"]).dt.days,
        (today - df["subscription_start_date"]).dt.days
    )

    conditions = [
        df["churn_score"] < 50,
        (df["churn_score"] >= 50) & (df["churn_score"] < 70),
        df["churn_score"] >= 70
    ]

    df["churn_risk"] = np.select(
        conditions,
        ["Low", "Medium", "High"],
        default="Unknown"
    )

    return df


try:
    df = load_data()

    # Sidebar filters
    st.sidebar.header("Filters")

    plans = st.sidebar.multiselect(
        "Select Plan Type",
        options=sorted(df["plan_type"].dropna().unique()),
        default=sorted(df["plan_type"].dropna().unique())
    )

    filtered_df = df[df["plan_type"].isin(plans)]

    # KPIs
    total_customers = len(filtered_df)
    churned_customers = filtered_df["churn_flag"].sum()

    churn_rate = (
        filtered_df["churn_flag"].mean() * 100
        if total_customers > 0 else 0
    )

    retention_rate = 100 - churn_rate
    arpu = filtered_df["monthly_charges"].mean()
    revenue_at_risk = filtered_df.loc[
        filtered_df["churn_flag"] == 1,
        "monthly_charges"
    ].sum()

    st.subheader("Business Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Customers", total_customers)
    col2.metric("Churned Customers", int(churned_customers))
    col3.metric("Churn Rate", f"{churn_rate:.2f}%")
    col4.metric("Retention Rate", f"{retention_rate:.2f}%")
    col5.metric("Revenue at Risk", f"₹{revenue_at_risk:,.2f}")

    st.divider()

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn by Plan Type")

        churn_by_plan = (
            filtered_df.groupby("plan_type")["churn_flag"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index(name="Churn Rate (%)")
        )

        st.bar_chart(
            churn_by_plan.set_index("plan_type")
        )

    with col2:
        st.subheader("Churn Risk Distribution")

        risk_count = (
            filtered_df["churn_risk"]
            .value_counts()
            .rename_axis("Risk")
            .to_frame("Customers")
        )

        st.bar_chart(risk_count)

    st.divider()

    # Customer details
    st.subheader("Customer Details")

    search = st.text_input(
        "Search by Customer ID or Name"
    )

    customer_view = filtered_df.copy()

    if search:
        customer_view = customer_view[
            customer_view["customerid"].astype(str).str.contains(
                search, case=False, na=False
            )
            | customer_view["customer_name"].astype(str).str.contains(
                search, case=False, na=False
            )
        ]

    st.dataframe(
        customer_view,
        use_container_width=True,
        hide_index=True
    )

except Exception as e:
    st.error("Error loading project data.")
    st.exception(e)