# ====================================================
# MarketMind AI Platform - Ultra Version
# AI Sales & Marketing Intelligence Dashboard
# ====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import google.generativeai as genai

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(page_title="MarketMind AI", layout="wide")

st.title("🚀 MarketMind AI Platform")
st.caption("AI Powered Sales & Marketing Intelligence")

# ---------------- SAMPLE DATA ----------------

data = {
"Month":["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
"Sales":[1200,1500,1700,2100,2500,2700,3000,3400,3600,4000,4300,4800],
"Marketing":[300,350,400,450,500,550,600,650,700,750,800,900],
"Customers":[100,130,150,180,220,260,300,320,340,380,420,460]
}

df = pd.DataFrame(data)

# ---------------- SIDEBAR ----------------

st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
"Select Module",
[
"Dashboard",
"Sales Forecasting",
"Customer Segmentation",
"Churn Prediction",
"Upload Dataset",
"AI Marketing Assistant"
]
)

# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.subheader("Business Overview")

    c1,c2,c3 = st.columns(3)

    c1.metric("Total Sales",df["Sales"].sum())
    c2.metric("Total Customers",df["Customers"].sum())
    c3.metric("Avg Marketing Spend",int(df["Marketing"].mean()))

    st.subheader("Sales Trend")

    fig,ax = plt.subplots()

    sns.lineplot(x="Month",y="Sales",data=df,marker="o",ax=ax)

    st.pyplot(fig)

# ---------------- SALES FORECASTING ----------------

elif page == "Sales Forecasting":

    st.header("Sales Prediction")

    X = df[["Marketing","Customers"]]
    y = df["Sales"]

    model = LinearRegression()

    model.fit(X,y)

    marketing = st.slider("Marketing Budget",100,2000,800)
    customers = st.slider("Expected Customers",50,1000,400)

    future = np.array([[marketing,customers]])

    prediction = model.predict(future)

    st.success(f"Predicted Sales: {int(prediction[0])}")

# ---------------- CUSTOMER SEGMENTATION ----------------

elif page == "Customer Segmentation":

    st.header("Customer Segmentation")

    customer_data = pd.DataFrame({
    "Age":[22,25,45,52,23,40,60,48,33,36],
    "Spending":[200,250,600,650,220,580,700,620,400,420]
    })

    scaler = StandardScaler()

    scaled = scaler.fit_transform(customer_data)

    kmeans = KMeans(n_clusters=3,n_init=10)

    customer_data["Segment"] = kmeans.fit_predict(scaled)

    st.dataframe(customer_data)

    fig2,ax2 = plt.subplots()

    sns.scatterplot(x="Age",y="Spending",hue="Segment",data=customer_data)

    st.pyplot(fig2)

# ---------------- CHURN PREDICTION ----------------

elif page == "Churn Prediction":

    st.header("Customer Churn Prediction")

    churn_data = pd.DataFrame({
    "Visits":[5,3,10,12,2,8,7,1,4,9],
    "Purchases":[2,1,5,6,1,4,3,0,2,4],
    "Churn":[1,1,0,0,1,0,0,1,1,0]
    })

    X = churn_data[["Visits","Purchases"]]
    y = churn_data["Churn"]

    rf = RandomForestClassifier()

    rf.fit(X,y)

    visits = st.slider("Customer Visits",1,15,3)
    purchases = st.slider("Purchases",0,10,1)

    test = pd.DataFrame([[visits,purchases]],columns=["Visits","Purchases"])

    prediction = rf.predict(test)

    if prediction[0] == 1:
        st.error("⚠ Customer may leave")
    else:
        st.success("Customer likely to stay")

# ---------------- DATASET UPLOAD ----------------

elif page == "Upload Dataset":

    st.header("Upload Dataset")

    file = st.file_uploader("Upload CSV",type=["csv"])

    if file:

        data = pd.read_csv(file)

        st.dataframe(data)

        st.write("Dataset Statistics")

        st.write(data.describe())

# ---------------- AI MARKETING ASSISTANT ----------------

elif page == "AI Marketing Assistant":

    st.header("AI Marketing Assistant")

    api_key = st.text_input("Enter Gemini API Key",type="password")

    prompt = st.text_area("Ask AI about marketing strategy")

    if st.button("Generate AI Strategy"):
        if api_key == "":
            st.warning("vck_1FOad6nc6D6yvR9PzUWNSoqVDpsqTfcXhgyhKKttZtRDD3eM6b3dAwIM")

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content(prompt)

        st.write(response.text)