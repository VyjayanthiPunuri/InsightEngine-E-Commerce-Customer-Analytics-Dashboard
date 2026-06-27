 📊 InsightEngine — E-Commerce Customer Analytics Platform

> An end-to-end data science project built on the UCI Online Retail II dataset, covering data cleaning, exploratory analysis, RFM-based customer segmentation, churn prediction, and a collaborative-filter recommendation engine — all served through an interactive Streamlit dashboard.

---

## 🗂️ Project Structure

```
├── 01_data_cleaning.ipynb          # Data ingestion & cleaning pipeline
├── 02_eda.ipynb                    # Exploratory Data Analysis
├── 03_customer_segmentation.ipynb  # K-Means RFM segmentation
├── 04_churn_prediction.ipynb       # Random Forest churn model
├── 05_recommendation_engine.ipynb  # Collaborative filtering engine
├── app.py                          # Streamlit dashboard (InsightEngine)
│
├── cleaned_data.csv                # Output of notebook 01
├── eda_summary.csv                 # Output of notebook 02
├── customer_segments.csv           # Output of notebook 03
├── customer_churn_predictions.csv  # Output of notebook 04
├── user_item_matrix.csv            # Output of notebook 05
│
├── kmeans_model.pkl                # Saved KMeans model
├── scaler.pkl                      # Saved StandardScaler
├── churn_model.pkl                 # Saved RandomForest model
└── customer_similarity.pkl         # Precomputed cosine similarity matrix
```

---

## 🚀 Dashboard Pages

| Page | Description |
|---|---|
| Executive Dashboard | 12 live KPIs + monthly revenue trend, top countries, weekday analysis |
| Analytics | Revenue, customer, product & geographic deep-dives with interactive filters |
| Segmentation | RFM cluster profiles, CLV distribution, bubble chart, business insights |
| Live Churn | Real-time churn probability via trained model + gauge chart + retention playbook |
| Recommendation | Collaborative-filter product recommendations per customer |
| Model Performance | Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, feature importance |

---

## 🔬 Notebooks Overview

### 01 — Data Cleaning
Loads the raw UCI Online Retail II Excel file, removes cancellations (invoices starting with `C`), drops rows with missing `Customer ID`, filters out zero/negative prices and quantities, engineers `Revenue = Quantity × Price`, and extracts `Year`, `Month`, `Day`, `Hour` from `InvoiceDate`. Output: `cleaned_data.csv`.

### 02 — EDA
Analyses revenue by time (monthly, hourly, weekday), identifies top countries and products, examines purchase frequency distribution and correlation structure. Summary statistics saved to `eda_summary.csv`.

### 03 — Customer Segmentation
Computes RFM (Recency, Frequency, Monetary) metrics per customer, adds `Total_Items`, `Average_Order_Value`, and `CLV`. Features are standardised and clustered using K-Means (Elbow + Silhouette methods for k selection). Clusters are labelled as business segments (`Champions`, `Loyal`, `High Value`, `At Risk`, `Low Value`). Saves `customer_segments.csv`, `kmeans_model.pkl`, `scaler.pkl`.

### 04 — Churn Prediction
Defines churn as customers with Recency > threshold. Trains a Random Forest classifier on RFM-derived features. Evaluates with accuracy, precision, recall, F1, ROC-AUC. Saves `customer_churn_predictions.csv` and `churn_model.pkl`.

### 05 — Recommendation Engine
Builds a customer × product interaction matrix. Computes cosine similarity between customers. For a given customer, finds the 5 most similar peers and recommends products they purchased but the target customer has not. Saves `user_item_matrix.csv` and `customer_similarity.pkl`.

---

## 📦 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/insightengine.git
cd insightengine
```

### 2. Create environment
```bash
conda create -n customer_env python=3.11
conda activate customer_env
pip install -r requirements.txt
```

### 3. Run notebooks in order
Open JupyterLab or Jupyter Notebook and run `01` → `02` → `03` → `04` → `05`. Each notebook generates the data/model files consumed by the next.

### 4. Launch the dashboard
```bash
streamlit run app.py
```

---

## 📋 Requirements

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
joblib
streamlit
openpyxl
```

---

## 📊 Dataset

**UCI Online Retail II** — Transactions from a UK-based online gift retailer (2009–2011).

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- Licence: Creative Commons Attribution 4.0

---

## 🎯 Key Results

| Metric | Value |
|---|---|
| Total Customers | ~5,878 |
| Churn Rate | ~40.8% |
| Customer Segments | 5 (Loyal, High Value, At Risk, Low Value, Champions) |
| Largest Segment | Loyal (3,799 customers) |
| Model | Random Forest Classifier |

---

## 💡 Skills Demonstrated

- Data wrangling and feature engineering with **Pandas / NumPy**
- Exploratory data analysis with **Matplotlib, Seaborn, Plotly**
- Unsupervised learning — **K-Means clustering**, Elbow & Silhouette
- Supervised learning — **Random Forest**, ROC-AUC, classification metrics
- Collaborative filtering — **cosine similarity** user-item matrix
- Dashboard development — **Streamlit** with multi-page layout and live predictions
- Model persistence — **Joblib** serialisation

---

## 👤 Author

**Vyjayanthi**  
BS-MS Computer Science & Business Management  
[LinkedIn](#) · [GitHub](#)
