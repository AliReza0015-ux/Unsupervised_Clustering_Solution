import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# Page title
st.title("Mall Customer Segmentation Model - Clustering Solution")

# Load data
df = pd.read_csv('mall_customers.csv')

st.header("Dataset Overview")
st.dataframe(df.head())
st.write(f"Dataset shape: {df.shape}")

st.header("Statistical Summary")
st.dataframe(df.describe())

st.header("Correlation Matrix (Pearson)")
corr_pearson = df.corr(numeric_only=True)
st.dataframe(corr_pearson)

st.header("Correlation Matrix (Spearman)")
corr_spearman = df.corr(numeric_only=True, method='spearman')
st.dataframe(corr_spearman)

st.header("Pairplot of Age, Annual Income and Spending Score")
fig1 = sns.pairplot(df[['Age', 'Annual_Income', 'Spending_Score']])
st.pyplot(fig1)

st.markdown("""
**Observations:**  
- Spending Score is high for customers between age 20-40.  
- K-means is sensitive to outliers, consider removing if needed.
""")

# Clustering with 2 features
st.header("K-Means Clustering on Annual Income and Spending Score")
kmodel = KMeans(n_clusters=5, random_state=42).fit(df[['Annual_Income', 'Spending_Score']])
df['Cluster'] = kmodel.labels_

st.subheader("Cluster Centers")
st.write(pd.DataFrame(kmodel.cluster_centers_, columns=['Annual_Income', 'Spending_Score']))

st.subheader("Cluster Counts")
st.write(df['Cluster'].value_counts())

st.subheader("Cluster Visualization")
fig2, ax2 = plt.subplots()
sns.scatterplot(x='Annual_Income', y='Spending_Score', data=df, hue='Cluster', palette='colorblind', ax=ax2)
st.pyplot(fig2)

# Elbow Method
st.header("Elbow Method to Find Optimal k")

# Step 1: Define k range
k_range = range(3, 9)
wcss = []

# Step 2: Loop to collect inertia values
for i in k_range:
    kmodel_i = KMeans(n_clusters=i, random_state=42)
    kmodel_i.fit(df[['Annual_Income', 'Spending_Score']])
    wcss.append(kmodel_i.inertia_)

# Step 3: Create DataFrame from results
wss_df = pd.DataFrame({'Clusters': list(k_range), 'WCSS': wcss})

# Step 4: Debug and display
st.write("DataFrame columns:", wss_df.columns)
st.write(wss_df.head())

# Step 5: Show line chart
st.line_chart(wss_df.set_index('Clusters'))


# Silhouette Score
st.header("Silhouette Score for various k")
sil_scores = []
for i in k_range:
    kmodel_i = KMeans(n_clusters=i, random_state=42).fit(df[['Annual_Income', 'Spending_Score']])
    labels = kmodel_i.labels_
    sil_scores.append(silhouette_score(df[['Annual_Income', 'Spending_Score']], labels))

sil_df = pd.DataFrame({'Clusters': list(k_range), 'Silhouette_Score': sil_scores})
st.line_chart(sil_df.rename(columns={'Clusters':'index'}).set_index('Clusters'))

st.markdown("""
**Conclusion:**  
Both Elbow and Silhouette methods suggest optimal clusters around k=5.
""")

# Using all 3 features now
st.header("K-Means Clustering with Age, Annual Income and Spending Score")

sil_scores_3f = []
for i in k_range:
    kmodel_i = KMeans(n_clusters=i, random_state=42).fit(df[['Age', 'Annual_Income', 'Spending_Score']])
    labels = kmodel_i.labels_
    sil_scores_3f.append(silhouette_score(df[['Age', 'Annual_Income', 'Spending_Score']], labels))

sil_3f_df = pd.DataFrame({'Clusters': list(k_range), 'Silhouette_Score': sil_scores_3f})
st.line_chart(sil_3f_df.rename(columns={'Clusters':'index'}).set_index('Clusters'))

# Elbow for 3 features
wcss_3f = []
for i in k_range:
    kmodel_i = KMeans(n_clusters=i, random_state=42).fit(df[['Annual_Income', 'Spending_Score', 'Age']])
    wcss_3f.append(kmodel_i.inertia_)

wss_3f_df = pd.DataFrame({'Clusters': list(k_range), 'WCSS': wcss_3f})
st.line_chart(wss_3f_df.rename(columns={'Clusters':'index'}).set_index('Clusters'))

st.markdown("""
**Conclusion:**  
With 3 features, the optimal number of clusters looks closer to k=6.
""")

st.markdown("""
### Exercise:  
Try training KMeans with `init='k-means++'` parameter to improve centroid initialization.
""")
