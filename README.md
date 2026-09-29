# 🔮 Enterprise Customer Churn & Risk Intelligence Dashboard

An end-to-end Machine Learning and Business Intelligence solution for predicting customer churn, identifying high-risk subscriber segments, and testing retention pricing strategies.

---

## 🚀 Key Features

1. **🤖 Machine Learning Churn Prediction**:
   - Gradient Boosting Classifier trained on customer demographics, plan details, support ticket history, CSAT scores, and tenure.
   - **99.7% Accuracy** and **0.995 ROC-AUC** score.
   - Real-time churn probability scoring (0-100%) for all active customers.

2. **📊 Executive Dashboard & Interactive KPIs**:
   - Built with Streamlit and Plotly graphics.
   - Key KPIs: Total Customers, Churn Rate %, Retention Rate %, ARPU, Total Revenue at Risk, Predicted High-Risk Revenue.
   - Multi-dimensional sidebar filters: Plan Type, Contract Type, Country, and ML Churn Risk Tier.

3. **👤 Individual Customer Scorecard & What-If Simulator**:
   - Interactive gauge chart for individual customer risk scores.
   - What-if strategy simulator: Test discount offers or plan upgrades in real time to see how they reduce churn probability.

4. **💬 Support & Feedback Intelligence**:
   - Analysis of support complaints, escalation rates, and CSAT scores.
   - Frequency breakdown of top customer complaint themes.

5. **📥 Export & Actionable Target Lists**:
   - Export filtered dataset to CSV.
   - Download High-Risk Customer Outreach List for Customer Success teams.

---

## 📁 Repository Structure

```
├── app.py                      # Main Streamlit dashboard application
├── train_model.py              # ML pipeline training script
├── generate_data.py            # Synthetic data generation script (2,000 customers)
├── churn_model.pkl             # Serialized ML model and preprocessor pipeline
├── customer_churn.db           # SQLite database (db_customer, db_subscription, db_support)
├── customer_churn_data_raw.xlsx# Excel source file
├── exported_churn_data.csv     # Exported combined CSV dataset
├── requirements.txt            # Project Python dependencies
└── README.md                   # Documentation
```

---

## ⚙️ Installation & Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train / Re-train ML Model (Optional)**:
   ```bash
   python3 train_model.py
   ```

3. **Launch Dashboard**:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` or `http://localhost:8502` in your web browser.
