# 🔮 Enterprise Customer Churn & Risk Intelligence Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=flat&logo=plotly&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite&logoColor=white)

> An **end-to-end Machine Learning + Business Intelligence** solution for predicting customer churn, identifying high-risk subscriber segments, quantifying revenue at risk, and simulating real-time retention strategies — all in an interactive enterprise-grade dashboard.

---

## 🎯 Problem Statement

Subscription-based businesses lose billions in revenue due to unexpected customer churn. This project builds a **production-ready churn intelligence platform** that:
- Predicts which customers are likely to cancel (before they do)
- Quantifies monthly revenue at risk per segment
- Enables customer success teams to simulate and act on retention strategies in real time

---

## 🚀 Key Features

### 🤖 ML Churn Prediction Engine
- **Gradient Boosting Classifier** trained on customer demographics, subscription details, support ticket history, CSAT scores, and tenure
- **99.7% Test Accuracy** | **0.995 ROC-AUC Score**
- Scikit-Learn `Pipeline` with `ColumnTransformer` (StandardScaler + OneHotEncoder)
- Real-time churn probability scoring (0–100%) per customer

### 📊 Executive Intelligence Dashboard
- Built with **Streamlit** and **Plotly** for interactive, real-time charts
- Key KPIs: Total Customers, Churn Rate %, Retention Rate %, ARPU, Revenue at Risk
- Multi-dimensional filters: Plan Type, Contract Type, Country, ML Risk Tier

### 👤 Customer Scorecard & What-If Simulator
- Individual customer risk gauge chart
- **Live What-If Simulator**: Test discount offers or plan/contract upgrades in real time to see the predicted change in churn probability

### 💬 Support & Feedback Intelligence
- Support ticket analytics: Escalation rate, avg CSAT score
- Top complaint theme frequency charts
- CSAT distribution breakdown by escalation status

### 📥 Actionable Data Exports
- Download filtered dataset as CSV
- One-click **High-Risk Customer Outreach List** for customer success teams

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | Scikit-Learn (Gradient Boosting, Pipeline, ColumnTransformer) |
| Data Processing | Pandas, NumPy |
| Database | SQLite (3 relational tables) |
| Dashboard | Streamlit |
| Visualizations | Plotly Express & Graph Objects |
| Data Source | Excel → SQLite ETL pipeline |

---

## 📁 Repository Structure

```
churn_analysis/
├── app.py                  # Main Streamlit dashboard (5 tabs, 15+ charts)
├── train_model.py          # ML training pipeline script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

> **Note:** `customer_churn.db`, `churn_model.pkl`, and raw data files are excluded from the repo via `.gitignore`. Run `train_model.py` to regenerate the model locally.

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/churn_analysis.git
cd churn_analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the ML model
```bash
python3 train_model.py
```

### 4. Launch the dashboard
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | **99.7%** |
| ROC-AUC Score | **0.995** |
| Algorithm | Gradient Boosting Classifier |
| Train/Test Split | 80/20 stratified |

### Top Churn Predictors (Feature Importance)
1. `tenure_days` — Customer longevity
2. `monthly_charges` — Pricing sensitivity
3. `csat_score` — Support satisfaction
4. `contract_type` — Monthly vs Annual lock-in
5. `complaint_count` — Support friction

---

## 💡 Business Impact

- Identified **high-risk customer segments** enabling proactive outreach before cancellation
- Quantified **monthly revenue at risk** (₹) per plan type and contract type
- Built a **what-if simulator** allowing business teams to model the financial impact of discount or upgrade offers without any code
- Enabled **data-driven retention campaigns** by exporting prioritized customer outreach lists

---

## 📬 Contact

**Siddharth Kushwaha**
- 📧 [your-email@example.com](mailto:your-email@example.com)
- 💼 [LinkedIn](https://linkedin.com/in/your-profile)
- 🐙 [GitHub](https://github.com/your-username)
