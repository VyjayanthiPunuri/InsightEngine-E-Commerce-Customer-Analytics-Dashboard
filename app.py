import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="InsightEngine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main{
    background-color:#f5f7fb;
}

.block-container{
    padding-top:1rem;
}

.metric-card{
    background:white;
    border-radius:12px;
    padding:18px;
    box-shadow:0px 3px 12px rgba(0,0,0,.12);
    transition:.3s;
}

.metric-card:hover{
    transform:scale(1.02);
}

h1,h2,h3{
    color:#1b2559;
}

[data-testid="stSidebar"]{
    background:#1b2559;
}

[data-testid="stSidebar"] *{
    color:white;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Data
# -----------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "cleaned_data.csv",
        parse_dates=["InvoiceDate"]
    )

    segments = pd.read_csv(
        "customer_segments.csv"
    )

    churn = pd.read_csv(
        "customer_churn_predictions.csv"
    )

    user_item = pd.read_csv(
        "user_item_matrix.csv",
        index_col=0
    )

    return df,segments,churn,user_item


df,segments,churn,user_item = load_data()

similarity = joblib.load(
    "customer_similarity.pkl"
)

churn_model = joblib.load(
    "churn_model.pkl"
)

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.image(
    "https://img.icons8.com/color/96/combo-chart--v1.png",
    width=70
)

st.sidebar.title("InsightEngine")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Analytics",
        "Segmentation",
        "Live Churn",
        "Recommendation",
        "Model Performance"
    ]
)

st.sidebar.markdown("---")

# -----------------------------
# Global Filters
# -----------------------------

country = st.sidebar.selectbox(
    "Country",
    ["All"] + sorted(df["Country"].unique())
)

year = st.sidebar.selectbox(
    "Year",
    ["All"] + sorted(df["Year"].unique().tolist())
)

month = st.sidebar.selectbox(
    "Month",
    ["All"] + sorted(df["Month"].unique().tolist())
)

filtered = df.copy()

if country != "All":
    filtered = filtered[
        filtered["Country"] == country
    ]

if year != "All":
    filtered = filtered[
        filtered["Year"] == year
    ]

if month != "All":
    filtered = filtered[
        filtered["Month"] == month
    ]

# -----------------------------
# Executive Dashboard
# -----------------------------

if page=="Executive Dashboard":

    st.title("📊 Executive Dashboard")

    st.markdown(
        "### E-Commerce Customer Analytics"
    )

    st.divider()

    # -----------------------------
    # KPIs
    # -----------------------------

    revenue = filtered["Revenue"].sum()

    orders = filtered["Invoice"].nunique()

    customers = filtered["Customer ID"].nunique()

    products = filtered["StockCode"].nunique()

    quantity = filtered["Quantity"].sum()

    countries = filtered["Country"].nunique()

    avg_order = (
        filtered.groupby("Invoice")["Revenue"]
        .sum()
        .mean()
    )

    avg_customer = (
        filtered.groupby("Customer ID")["Revenue"]
        .sum()
        .mean()
    )

    repeat = (
        filtered.groupby("Customer ID")["Invoice"]
        .nunique()
    )

    repeat_rate = (
        (repeat>1).sum()/len(repeat)
    )*100

    churn_rate = (
        churn["Churn"].mean()
    )*100

    clv = (
        segments["CLV"].mean()
    )

    top_customer = (
        filtered.groupby("Customer ID")["Revenue"]
        .sum()
        .max()
    )

    # -----------------------------
    # KPI ROW 1
    # -----------------------------

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "💰 Revenue",
        f"${revenue:,.0f}"
    )

    c2.metric(
        "🛒 Orders",
        f"{orders:,}"
    )

    c3.metric(
        "👥 Customers",
        f"{customers:,}"
    )

    c4.metric(
        "📦 Products",
        f"{products:,}"
    )

    # -----------------------------
    # KPI ROW 2
    # -----------------------------

    c5,c6,c7,c8 = st.columns(4)

    c5.metric(
        "📦 Quantity",
        f"{quantity:,}"
    )

    c6.metric(
        "🌍 Countries",
        countries
    )

    c7.metric(
        "💳 Avg Order",
        f"${avg_order:.2f}"
    )

    c8.metric(
        "👤 Avg Customer",
        f"${avg_customer:.2f}"
    )

    # -----------------------------
    # KPI ROW 3
    # -----------------------------

    c9,c10,c11,c12 = st.columns(4)

    c9.metric(
        "🔄 Repeat %",
        f"{repeat_rate:.1f}%"
    )

    c10.metric(
        "⚠️ Churn %",
        f"{churn_rate:.1f}%"
    )

    c11.metric(
        "💎 Avg CLV",
        f"${clv:,.0f}"
    )

    c12.metric(
        "🏆 Top Customer",
        f"${top_customer:,.0f}"
    )

    st.divider()

    # -----------------------------
    # Monthly Revenue
    # -----------------------------

    monthly = (
        filtered.groupby(
            filtered["InvoiceDate"].dt.to_period("M")
        )["Revenue"]
        .sum()
        .reset_index()
    )

    monthly["InvoiceDate"] = (
        monthly["InvoiceDate"].astype(str)
    )

    fig = px.line(
        monthly,
        x="InvoiceDate",
        y="Revenue",
        markers=True,
        title="Monthly Revenue Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------
    # Two Charts
    # -----------------------------

    col1,col2 = st.columns(2)

    with col1:

        country_sales = (
            filtered.groupby("Country")["Revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            country_sales,
            x="Country",
            y="Revenue",
            color="Revenue",
            title="Top 10 Countries"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        weekday = (
            filtered.groupby("Day")["Revenue"]
            .sum()
            .reindex([
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ])
            .reset_index()
        )

        fig = px.bar(
            weekday,
            x="Day",
            y="Revenue",
            color="Revenue",
            title="Revenue by Weekday"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------
    # Expandable Reports
    # -----------------------------

    with st.expander(
        "📈 Revenue Analysis"
    ):

        hourly = (
            filtered.groupby("Hour")["Revenue"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            hourly,
            x="Hour",
            y="Revenue",
            markers=True,
            title="Revenue by Hour"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig = px.histogram(
            filtered,
            x="Revenue",
            nbins=50,
            title="Revenue Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        top_products = (
            filtered.groupby("Description")["Revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            top_products,
            x="Revenue",
            y="Description",
            orientation="h",
            title="Top Revenue Products"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
# ======================================================
# ANALYTICS DASHBOARD
# ======================================================

elif page == "Analytics":

    st.title("📈 Analytics Dashboard")
    st.markdown("### Interactive Business Analytics")
    st.divider()

    # -------------------------
    # KPI Summary
    # -------------------------

    k1,k2,k3,k4 = st.columns(4)

    k1.metric(
        "Revenue",
        f"${filtered['Revenue'].sum():,.0f}"
    )

    k2.metric(
        "Orders",
        filtered["Invoice"].nunique()
    )

    k3.metric(
        "Customers",
        filtered["Customer ID"].nunique()
    )

    k4.metric(
        "Products",
        filtered["Description"].nunique()
    )

    st.divider()

    # ==================================================
    # Revenue Analytics
    # ==================================================

    with st.expander("📈 Revenue Analytics", expanded=True):

        col1,col2 = st.columns(2)

        with col1:

            monthly = (
                filtered.groupby("Month")["Revenue"]
                .sum()
                .reset_index()
            )

            fig = px.bar(
                monthly,
                x="Month",
                y="Revenue",
                color="Revenue",
                title="Revenue by Month"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            hourly = (
                filtered.groupby("Hour")["Revenue"]
                .sum()
                .reset_index()
            )

            fig = px.line(
                hourly,
                x="Hour",
                y="Revenue",
                markers=True,
                title="Revenue by Hour"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        fig = px.histogram(
            filtered,
            x="Revenue",
            nbins=40,
            title="Revenue Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ==================================================
    # Customer Analytics
    # ==================================================

    with st.expander("👥 Customer Analytics"):

        top_customers = (
            filtered.groupby("Customer ID")["Revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            top_customers,
            x="Customer ID",
            y="Revenue",
            color="Revenue",
            title="Top 10 Customers"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        purchase_frequency = (
            filtered.groupby("Customer ID")["Invoice"]
            .nunique()
            .reset_index(name="Orders")
        )

        fig = px.histogram(
            purchase_frequency,
            x="Orders",
            nbins=25,
            title="Customer Purchase Frequency"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ==================================================
    # Product Analytics
    # ==================================================

    with st.expander("📦 Product Analytics"):

        top_products = (
            filtered.groupby("Description")["Revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            top_products,
            x="Revenue",
            y="Description",
            orientation="h",
            color="Revenue",
            title="Top Revenue Products"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        quantity = (
            filtered.groupby("Description")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            quantity,
            x="Quantity",
            y="Description",
            orientation="h",
            color="Quantity",
            title="Top Selling Products"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        treemap = (
            filtered.groupby("Description")["Revenue"]
            .sum()
            .reset_index()
            .sort_values("Revenue", ascending=False)
            .head(25)
        )

        fig = px.treemap(
            treemap,
            path=["Description"],
            values="Revenue",
            title="Revenue Treemap"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ==================================================
    # Geographic Analysis
    # ==================================================

    with st.expander("🌍 Geographic Analysis"):

        geo = (
            filtered.groupby("Country")
            .agg({
                "Revenue":"sum",
                "Invoice":"nunique",
                "Customer ID":"nunique"
            })
            .reset_index()
        )

        fig = px.scatter(
            geo,
            x="Customer ID",
            y="Revenue",
            size="Invoice",
            color="Revenue",
            hover_name="Country",
            title="Country Performance"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ==================================================
    # Data Quality
    # ==================================================

    with st.expander("📋 Dataset Summary"):

        c1,c2,c3 = st.columns(3)

        c1.metric(
            "Rows",
            f"{len(filtered):,}"
        )

        c2.metric(
            "Columns",
            filtered.shape[1]
        )

        c3.metric(
            "Missing Values",
            int(filtered.isnull().sum().sum())
        )

        st.dataframe(
            filtered.describe(include="all")
        )

    # ==================================================
    # Correlation Analysis
    # ==================================================

    with st.expander("📊 Correlation Analysis"):

        corr = filtered[
            [
                "Quantity",
                "Price",
                "Revenue"
            ]
        ].corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig = px.scatter(
            filtered.sample(min(5000,len(filtered))),
            x="Quantity",
            y="Revenue",
            color="Price",
            title="Quantity vs Revenue"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
# ======================================================
# CUSTOMER SEGMENTATION
# ======================================================

elif page == "Segmentation":

    st.title("🎯 Customer Segmentation Dashboard")
    st.markdown("### RFM Analysis & Customer Intelligence")
    st.divider()

    # --------------------------------------------------
    # KPIs
    # --------------------------------------------------

    total_segments = segments["Segment"].nunique()
    avg_clv = segments["CLV"].mean()
    avg_frequency = segments["Frequency"].mean()
    avg_monetary = segments["Monetary"].mean()

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Customer Segments",
        total_segments
    )

    c2.metric(
        "Average CLV",
        f"${avg_clv:,.2f}"
    )

    c3.metric(
        "Avg Purchase Frequency",
        f"{avg_frequency:.2f}"
    )

    c4.metric(
        "Avg Monetary Value",
        f"${avg_monetary:,.2f}"
    )

    st.divider()

    # --------------------------------------------------
    # Segment Distribution
    # --------------------------------------------------

    with st.expander(
        "📊 Segment Distribution",
        expanded=True
    ):

        segment_count = (
            segments["Segment"]
            .value_counts()
            .reset_index()
        )

        segment_count.columns = [
            "Segment",
            "Customers"
        ]

        fig = px.pie(
            segment_count,
            names="Segment",
            values="Customers",
            hole=0.45,
            title="Customer Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------
    # Revenue Contribution
    # --------------------------------------------------

    with st.expander(
        "💰 Revenue Contribution by Segment"
    ):

        revenue_segment = (
            segments.groupby("Segment")["Monetary"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            revenue_segment,
            x="Segment",
            y="Monetary",
            color="Monetary",
            title="Revenue Contribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------
    # CLV
    # --------------------------------------------------

    with st.expander(
        "💎 Customer Lifetime Value"
    ):

        fig = px.box(
            segments,
            x="Segment",
            y="CLV",
            color="Segment",
            title="CLV Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------
    # Segment Profile
    # --------------------------------------------------

    with st.expander(
        "📋 Segment Profile"
    ):

        profile = (
            segments.groupby("Segment")[
                [
                    "Recency",
                    "Frequency",
                    "Monetary",
                    "Average_Order_Value",
                    "CLV"
                ]
            ]
            .mean()
            .round(2)
        )

        st.dataframe(
            profile,
            use_container_width=True
        )

    # --------------------------------------------------
    # Segment Comparison
    # --------------------------------------------------

    with st.expander(
        "📈 Segment Comparison"
    ):

        compare = (
            segments.groupby("Segment")[
                [
                    "Recency",
                    "Frequency",
                    "Monetary"
                ]
            ]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            compare,
            x="Segment",
            y="Monetary",
            color="Segment",
            title="Average Monetary Value"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig = px.bar(
            compare,
            x="Segment",
            y="Frequency",
            color="Segment",
            title="Average Purchase Frequency"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------
    # Bubble Chart
    # --------------------------------------------------

    with st.expander(
        "🫧 Customer Bubble Chart"
    ):

        fig = px.scatter(
            segments,
            x="Frequency",
            y="Monetary",
            size="CLV",
            color="Segment",
            hover_data=["Recency"],
            title="Frequency vs Monetary"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------
    # Top Customers
    # --------------------------------------------------

    with st.expander(
        "🏆 Top Customers"
    ):

        top = (
            segments.sort_values(
                "CLV",
                ascending=False
            )
            .head(20)
        )

        st.dataframe(
            top,
            use_container_width=True
        )

    # --------------------------------------------------
    # Segment Insights
    # --------------------------------------------------

    with st.expander(
        "📑 Business Insights",
        expanded=True
    ):

        highest = (
            segments.groupby("Segment")["CLV"]
            .mean()
            .idxmax()
        )

        lowest = (
            segments.groupby("Segment")["CLV"]
            .mean()
            .idxmin()
        )

        st.success(f"""

### Executive Summary

🏆 Highest Value Segment : **{highest}**

⚠ Lowest Value Segment : **{lowest}**

Average Customer Lifetime Value : **${avg_clv:,.2f}**

Average Purchase Frequency : **{avg_frequency:.2f}**

Average Customer Spend : **${avg_monetary:,.2f}**

---

### Recommendations

✅ Launch loyalty campaigns for high-value customers.

✅ Retarget customers with high Recency.

✅ Increase cross-selling for medium-value customers.

✅ Focus marketing budget on the highest-performing customer segment.

""")
# ======================================================
# LIVE CHURN PREDICTION
# ======================================================

elif page == "Live Churn":

    st.title("🤖 Live Customer Churn Prediction")
    st.markdown("### Predict Customer Churn Risk in Real Time")
    st.divider()

    st.info(
        "Enter the customer details below to estimate churn probability and receive retention recommendations."
    )

    # -----------------------------
    # Input Section
    # -----------------------------

    col1, col2 = st.columns(2)

    with col1:

        recency = st.number_input(
            "Recency (Days)",
            min_value=0,
            value=30
        )

        frequency = st.number_input(
            "Purchase Frequency",
            min_value=1,
            value=5
        )

        monetary = st.number_input(
            "Monetary Value ($)",
            min_value=0.0,
            value=500.0
        )

    with col2:

        total_items = st.number_input(
            "Total Items Purchased",
            min_value=1,
            value=25
        )

        avg_order = st.number_input(
            "Average Order Value",
            min_value=1.0,
            value=100.0
        )

        clv = st.number_input(
            "Customer Lifetime Value",
            min_value=1.0,
            value=2500.0
        )

    st.divider()

    # -----------------------------
    # Prediction
    # -----------------------------

    if st.button("🚀 Predict Churn"):

        sample = pd.DataFrame(
            [[
                recency,
                frequency,
                monetary,
                total_items,
                avg_order,
                clv
            ]],
            columns=[
                "Recency",
                "Frequency",
                "Monetary",
                "Total_Items",
                "Average_Order_Value",
                "CLV"
            ]
        )

        probability = churn_model.predict_proba(sample)[0][1]

        prediction = churn_model.predict(sample)[0]

        st.subheader("Prediction Result")

        c1,c2,c3 = st.columns(3)

        c1.metric(
            "Churn Probability",
            f"{probability*100:.2f}%"
        )

        c2.metric(
            "Prediction",
            "Churn" if prediction==1 else "Active"
        )

        if probability < 0.30:

            risk = "🟢 LOW"

        elif probability < 0.70:

            risk = "🟡 MEDIUM"

        else:

            risk = "🔴 HIGH"

        c3.metric(
            "Risk",
            risk
        )

        # -------------------------
        # Gauge
        # -------------------------

        gauge = go.Figure(
            go.Indicator(

                mode="gauge+number",

                value=probability*100,

                title={"text":"Customer Churn Risk"},

                gauge={

                    "axis":{"range":[0,100]},

                    "bar":{"color":"darkblue"},

                    "steps":[

                        {"range":[0,30],"color":"green"},

                        {"range":[30,70],"color":"gold"},

                        {"range":[70,100],"color":"red"}

                    ]

                }

            )
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        # -------------------------
        # Recommendations
        # -------------------------

        st.subheader("Business Recommendation")

        if probability < 0.30:

            st.success("""

### 🟢 Low Risk Customer

✅ Continue regular engagement.

✅ Recommend premium products.

✅ Encourage referrals.

✅ Cross-sell complementary products.

""")

        elif probability < 0.70:

            st.warning("""

### 🟡 Medium Risk Customer

⚠ Send personalized email campaign.

⚠ Offer loyalty rewards.

⚠ Recommend discounted bundles.

⚠ Follow-up within 30 days.

""")

        else:

            st.error("""

### 🔴 High Risk Customer

🚨 Immediate retention campaign.

🚨 Offer exclusive discount.

🚨 Assign relationship manager.

🚨 Send win-back campaign.

🚨 Personalized recommendations.

""")

        st.divider()

        # -------------------------
        # Feature Summary
        # -------------------------

        summary = pd.DataFrame({

            "Feature":[

                "Recency",
                "Frequency",
                "Monetary",
                "Items",
                "Average Order",
                "CLV"

            ],

            "Value":[

                recency,
                frequency,
                monetary,
                total_items,
                avg_order,
                clv

            ]

        })

        st.subheader("Customer Input Summary")

        st.dataframe(
            summary,
            use_container_width=True
        )

    st.divider()

    # -----------------------------
    # Existing Churn Dashboard
    # -----------------------------

    with st.expander(
        "📊 Existing Customer Churn Analysis",
        expanded=True
    ):

        high = churn[
            churn["Churn_Probability"] > 0.80
        ]

        medium = churn[
            (churn["Churn_Probability"] >= 0.30) &
            (churn["Churn_Probability"] <= 0.80)
        ]

        low = churn[
            churn["Churn_Probability"] < 0.30
        ]

        k1,k2,k3 = st.columns(3)

        k1.metric(
            "🔴 High Risk",
            len(high)
        )

        k2.metric(
            "🟡 Medium Risk",
            len(medium)
        )

        k3.metric(
            "🟢 Low Risk",
            len(low)
        )

        fig = px.histogram(
            churn,
            x="Churn_Probability",
            nbins=30,
            color="Churn",
            title="Distribution of Churn Probability"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("High Risk Customers")

        st.dataframe(
            high.head(20),
            use_container_width=True
        )

        st.download_button(
            "📥 Download High Risk Customers",
            high.to_csv(index=False),
            "High_Risk_Customers.csv",
            "text/csv"
        )
# ======================================================
# RECOMMENDATION ENGINE
# ======================================================

elif page == "Recommendation":

    st.title("🎁 AI Product Recommendation Engine")
    st.markdown("### Personalized Customer Recommendations")
    st.divider()

    # -----------------------------------------
    # Customer Selection
    # -----------------------------------------

    customer = st.selectbox(
        "Select Customer ID",
        sorted(user_item.index)
    )

    st.divider()

    # -----------------------------------------
    # Customer Purchase Summary
    # -----------------------------------------

    customer_orders = df[
        df["Customer ID"] == customer
    ]

    total_spent = customer_orders["Revenue"].sum()

    total_orders = customer_orders["Invoice"].nunique()

    total_products = customer_orders["StockCode"].nunique()

    avg_order = (
        total_spent / total_orders
        if total_orders > 0 else 0
    )

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Total Spend",
        f"${total_spent:,.2f}"
    )

    c2.metric(
        "Orders",
        total_orders
    )

    c3.metric(
        "Products Purchased",
        total_products
    )

    c4.metric(
        "Average Order",
        f"${avg_order:,.2f}"
    )

    st.divider()

    # -----------------------------------------
    # Purchased Products
    # -----------------------------------------

    with st.expander(
        "🛒 Previously Purchased Products",
        expanded=True
    ):

        purchased = customer_orders[
            [
                "Description",
                "Quantity",
                "Revenue"
            ]
        ]

        st.dataframe(
            purchased,
            use_container_width=True
        )

    # -----------------------------------------
    # Similar Customers
    # -----------------------------------------

    with st.expander(
        "👥 Similar Customers"
    ):

        similarity_scores = similarity.loc[customer]

        similar_customers = (
            similarity_scores
            .sort_values(
                ascending=False
            )
            .iloc[1:6]
        )

        similar_df = pd.DataFrame({

            "Customer ID":similar_customers.index,

            "Similarity Score":similar_customers.values

        })

        st.dataframe(
            similar_df,
            use_container_width=True
        )

        fig = px.bar(

            similar_df,

            x="Customer ID",

            y="Similarity Score",

            color="Similarity Score",

            title="Top Similar Customers"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------------------
    # Recommendation Logic
    # -----------------------------------------

    with st.expander(
        "🎯 Recommended Products",
        expanded=True
    ):

        customer_products = set(

            user_item.loc[customer]

            [user_item.loc[customer] > 0]

            .index

        )

        recommendations = {}

        for user in similar_customers.index:

            products = user_item.loc[user]

            for product in products[
                products > 0
            ].index:

                if product not in customer_products:

                    recommendations[product] = (

                        recommendations.get(
                            product,
                            0
                        ) + 1

                    )

        recommendations = sorted(

            recommendations.items(),

            key=lambda x:x[1],

            reverse=True

        )[:5]

        recommendation_codes = [

            x[0]

            for x in recommendations

        ]

        recommendation_table = df[
            df["StockCode"].isin(
                recommendation_codes
            )
        ][
            [
                "StockCode",
                "Description",
                "Price"
            ]
        ].drop_duplicates()

        st.dataframe(
            recommendation_table,
            use_container_width=True
        )

        fig = px.bar(

            recommendation_table,

            x="StockCode",

            y="Price",

            color="Price",

            hover_data=["Description"],

            title="Recommended Products"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------------------
    # Cross Sell
    # -----------------------------------------

    with st.expander(
        "📦 Cross Sell Opportunities"
    ):

        top_products = (

            customer_orders.groupby(

                "Description"

            )["Revenue"]

            .sum()

            .sort_values(

                ascending=False

            )

            .head(10)

            .reset_index()

        )

        fig = px.bar(

            top_products,

            x="Revenue",

            y="Description",

            orientation="h",

            color="Revenue",

            title="Customer Favourite Products"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    # -----------------------------------------
    # Business Insights
    # -----------------------------------------

    with st.expander(

        "📑 Recommendation Insights",

        expanded=True

    ):

        st.success(f"""

### Customer Summary

Customer ID : **{customer}**

Total Spend : **${total_spent:,.2f}**

Orders : **{total_orders}**

Products Purchased : **{total_products}**

---

### Recommendation Strategy

✅ Recommend products purchased by similar customers.

✅ Cross-sell products frequently bought together.

✅ Offer bundle discounts.

✅ Promote premium products based on purchase history.

✅ Send personalized email recommendations.

""")

    # -----------------------------------------
    # Download
    # -----------------------------------------

    csv = recommendation_table.to_csv(

        index=False

    )

    st.download_button(

        "📥 Download Recommendations",

        csv,

        "Recommended_Products.csv",

        "text/csv"

    )
# ======================================================
# MODEL PERFORMANCE
# ======================================================

elif page == "Model Performance":

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        ConfusionMatrixDisplay
    )

    st.title("📈 Machine Learning Model Performance")
    st.markdown("### Churn Prediction & Customer Segmentation")
    st.divider()

    # -------------------------------------------------
    # Recreate metrics from prediction file
    # -------------------------------------------------

    if "Churn" in churn.columns and "Churn_Probability" in churn.columns:

        y_true = churn["Churn"]

        y_pred = (
            churn["Churn_Probability"] >= 0.5
        ).astype(int)

        accuracy = accuracy_score(y_true,y_pred)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        roc = roc_auc_score(
            y_true,
            churn["Churn_Probability"]
        )

    else:

        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        roc = 0

    # -------------------------------------------------
    # KPI Cards
    # -------------------------------------------------

    c1,c2,c3,c4,c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        f"{accuracy:.3f}"
    )

    c2.metric(
        "Precision",
        f"{precision:.3f}"
    )

    c3.metric(
        "Recall",
        f"{recall:.3f}"
    )

    c4.metric(
        "F1 Score",
        f"{f1:.3f}"
    )

    c5.metric(
        "ROC AUC",
        f"{roc:.3f}"
    )

    st.divider()

    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------

    with st.expander(
        "📊 Confusion Matrix",
        expanded=True
    ):

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(
                x="Predicted",
                y="Actual"
            ),
            title="Confusion Matrix"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -------------------------------------------------
    # Feature Importance
    # -------------------------------------------------

    with st.expander(
        "🌲 Feature Importance"
    ):

        if hasattr(churn_model,"feature_importances_"):

            features = [

                "Recency",

                "Frequency",

                "Monetary",

                "Total Items",

                "Average Order",

                "CLV"

            ]

            importance = pd.DataFrame({

                "Feature":features,

                "Importance":churn_model.feature_importances_

            })

            importance = importance.sort_values(

                "Importance",

                ascending=False

            )

            fig = px.bar(

                importance,

                x="Importance",

                y="Feature",

                orientation="h",

                color="Importance",

                title="Random Forest Feature Importance"

            )

            st.plotly_chart(

                fig,

                use_container_width=True

            )

        else:

            st.info("Feature importance unavailable.")

    # -------------------------------------------------
    # Segment Distribution
    # -------------------------------------------------

    with st.expander(

        "👥 Customer Segment Distribution"

    ):

        seg = (

            segments["Segment"]

            .value_counts()

            .reset_index()

        )

        seg.columns = [

            "Segment",

            "Customers"

        ]

        fig = px.pie(

            seg,

            names="Segment",

            values="Customers",

            hole=.45,

            title="Customer Segments"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    # -------------------------------------------------
    # CLV Distribution
    # -------------------------------------------------

    with st.expander(

        "💎 Customer Lifetime Value"

    ):

        fig = px.histogram(

            segments,

            x="CLV",

            nbins=40,

            color="Segment",

            title="CLV Distribution"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    # -------------------------------------------------
    # Churn Probability
    # -------------------------------------------------

    with st.expander(

        "⚠ Customer Churn Probability"

    ):

        fig = px.histogram(

            churn,

            x="Churn_Probability",

            nbins=30,

            color="Churn",

            title="Churn Probability Distribution"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    # -------------------------------------------------
    # Model Summary
    # -------------------------------------------------

    with st.expander(

        "📑 Executive Model Summary",

        expanded=True

    ):

        best_feature = "N/A"

        if hasattr(churn_model,"feature_importances_"):

            best_feature = features[

                np.argmax(

                    churn_model.feature_importances_

                )

            ]

        st.success(f"""

# Machine Learning Summary

### Churn Model

✔ Accuracy : **{accuracy:.2%}**

✔ Precision : **{precision:.2%}**

✔ Recall : **{recall:.2%}**

✔ F1 Score : **{f1:.2%}**

✔ ROC AUC : **{roc:.2%}**

---

### Segmentation

Total Customer Segments :

**{segments['Segment'].nunique()}**

Average CLV :

**${segments['CLV'].mean():,.2f}**

Average Purchase Frequency :

**{segments['Frequency'].mean():.2f}**

Highest Impact Feature :

**{best_feature}**

---

### Business Recommendations

✅ Focus retention on customers with high Recency.

✅ Increase cross-selling for Loyal Customers.

✅ Launch premium offers for Champions.

✅ Monitor High-Risk customers weekly.

""")

    # -------------------------------------------------
    # Download Reports
    # -------------------------------------------------

    st.divider()

    st.subheader("📥 Download Reports")

    col1,col2,col3 = st.columns(3)

    with col1:

        st.download_button(

            "Customer Segments",

            segments.to_csv(index=False),

            "Customer_Segments.csv",

            "text/csv"

        )

    with col2:

        st.download_button(

            "Churn Predictions",

            churn.to_csv(index=False),

            "Customer_Churn.csv",

            "text/csv"

        )

    with col3:

        summary = pd.DataFrame({

            "Metric":[

                "Accuracy",

                "Precision",

                "Recall",

                "F1",

                "ROC AUC"

            ],

            "Value":[

                accuracy,

                precision,

                recall,

                f1,

                roc

            ]

        })

        st.download_button(

            "Model Summary",

            summary.to_csv(index=False),

            "Model_Performance.csv",

            "text/csv"

        )