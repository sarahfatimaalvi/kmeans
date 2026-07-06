import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ===========================
# Page Configuration
# ===========================

st.set_page_config(
    page_title="K-Means Clustering",
    page_icon="📊",
    layout="centered"
)

st.title("📊 K-Means Clustering")
st.write("Customer Segmentation using K-Means Clustering")

# ===========================
# Load Dataset
# ===========================

df = pd.read_csv("income.csv")

st.subheader("📋 Dataset")
st.dataframe(df)

# ===========================
# Train Model
# ===========================

X = df[['Age', 'Income($)']]

model = KMeans(n_clusters=3, random_state=42)

df['Cluster'] = model.fit_predict(X)

# ===========================
# User Input
# ===========================

st.subheader("📝 Enter Customer Details")

age = st.number_input(
    "Enter Age",
    min_value=1,
    max_value=100,
    value=30,
    step=1
)

income = st.number_input(
    "Enter Income ($)",
    min_value=1000,
    max_value=200000,
    value=50000,
    step=1000
)

# ===========================
# Prediction
# ===========================

if st.button("Predict Cluster"):

    cluster = model.predict([[age, income]])

    st.success(f"✅ This customer belongs to Cluster {cluster[0]}")

# ===========================
# Visualization
# ===========================

st.subheader("📊 K-Means Clustering Visualization")

fig, ax = plt.subplots(figsize=(8,6))

# Plot data points
scatter = ax.scatter(
    df['Age'],
    df['Income($)'],
    c=df['Cluster'],
    cmap='viridis',
    s=80
)

# Plot cluster centers
ax.scatter(
    model.cluster_centers_[:,0],
    model.cluster_centers_[:,1],
    color='red',
    marker='*',
    s=300,
    label='Centroids'
)

ax.set_xlabel("Age")
ax.set_ylabel("Income ($)")
ax.set_title("Customer Segmentation using K-Means")
ax.legend()

st.pyplot(fig)

# ===========================
# Cluster Centers
# ===========================

st.subheader("📍 Cluster Centers")

centers = pd.DataFrame(
    model.cluster_centers_,
    columns=['Age', 'Income ($)']
)

st.dataframe(centers)
