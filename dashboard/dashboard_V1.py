import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load preprocessed RFM data (replace with your actual CSV path or df)
df = pd.read_csv('data/rfm_clustered_data.csv')  # <- Ganti path jika perlu

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("📊 Customer Segmentation Dashboard (RFM + Clustering)")

# Sidebar filter
st.sidebar.header("🔍 Filters")
selected_cluster = st.sidebar.multiselect("Select Cluster(s):", sorted(df['Cluster'].unique()), default=sorted(df['Cluster'].unique()))

filtered_df = df[df['Cluster'].isin(selected_cluster)]

# Metric Cards
st.markdown("## 🔢 RFM Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Avg Recency", f"{filtered_df['Recency_RFM'].mean():.1f} days")
col2.metric("Avg Frequency", f"{filtered_df['Frequency_RFM'].mean():.1f}")
col3.metric("Avg Monetary", f"${filtered_df['Monetary_RFM'].mean():,.2f}")

# Cluster Distribution
st.markdown("---")
st.markdown("## 📌 Cluster Distribution")
fig1 = px.histogram(filtered_df, x="Cluster", color="Cluster", barmode="group",
                    title="Number of Customers per Cluster", text_auto=True)
st.plotly_chart(fig1, use_container_width=True)

# Boxplot Spending by Cluster
st.markdown("## 💰 Spending Distribution per Cluster")
fig2, ax = plt.subplots(figsize=(10, 4))
sns.boxplot(data=filtered_df, x='Cluster', y='Monetary_RFM', palette='Set3', ax=ax)
ax.set_title("Monetary Value Distribution by Cluster")
st.pyplot(fig2)

# Scatterplot Recency vs Monetary
st.markdown("## 🧭 Recency vs Monetary")
fig3 = px.scatter(filtered_df, x='Recency_RFM', y='Monetary_RFM', color='Cluster', size='Frequency_RFM',
                  title='Customer Distribution by Recency and Monetary', hover_data=['RFM_Score'])
st.plotly_chart(fig3, use_container_width=True)

# Segment Table
st.markdown("## 🧩 Segment Summary Table")
st.dataframe(filtered_df.groupby('Cluster')[['Recency_RFM', 'Frequency_RFM', 'Monetary_RFM']]
             .mean().round(2).rename(columns={
                 'Recency_RFM': 'Avg Recency',
                 'Frequency_RFM': 'Avg Frequency',
                 'Monetary_RFM': 'Avg Monetary'
             }))
