import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------
# Header
# --------------------------------

st.markdown("""
<h1 style='text-align: center;'>📦 Wholesale Customer Segmentation Dashboard</h1>
<p style='text-align: center;'>Interactive ML-based Customer Clustering System</p>
""", unsafe_allow_html=True)

# --------------------------------
# Sidebar Upload & Feature Selection
# --------------------------------

st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset Loaded Successfully")
else:
    try:
        df = pd.read_csv("Wholesale customers data.csv")
        st.sidebar.info("Using Default Dataset")
    except:
        st.error("Please upload a CSV file.")
        st.stop()

st.sidebar.header("⚙ Feature Selection")
numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
if 'Cluster' in numeric_cols: numeric_cols.remove('Cluster')

selected_features = st.sidebar.multiselect(
    "Select Features For Clustering",
    numeric_cols,
    default=numeric_cols[:3] if len(numeric_cols) > 3 else numeric_cols
)

if len(selected_features) < 2:
    st.warning("Please select at least TWO features")
    st.stop()

# --------------------------------
# Data Processing
# --------------------------------

X = df[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------
# SMALLER Elbow Method Plot
# --------------------------------

st.subheader("📈 Optimal Cluster Identification (Elbow Method)")

inertia = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# Creating a 3-column layout to center the plot at 50% width
e_col1, e_col2, e_col3 = st.columns([1, 2, 1])

with e_col2:
    # Small internal figsize
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(K_range, inertia, marker='o', linestyle='--', markersize=4)
    ax1.set_xlabel("Number of Clusters (K)", fontsize=9)
    ax1.set_ylabel("Inertia", fontsize=9)
    ax1.set_title("Elbow Curve", fontsize=10)
    ax1.tick_params(labelsize=8)
    ax1.grid(True, alpha=0.2)
    st.pyplot(fig1, use_container_width=True)

# --------------------------------
# Model Control
# --------------------------------

st.sidebar.header("🎯 Model Control")
k_value = st.sidebar.slider("Select Number of Clusters (K)", 2, 8, 3)

kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# --------------------------------
# SMALLER Cluster Visualization
# --------------------------------

st.subheader("🧭 Cluster Visualization")

v1, v2 = st.columns(2)
with v1:
    x_axis = st.selectbox("Select X Axis", selected_features, index=0)
with v2:
    y_axis = st.selectbox("Select Y Axis", selected_features, index=1 if len(selected_features) > 1 else 0)

# Centering the scatter plot at 50% width
v_col1, v_col2, v_col3 = st.columns([1, 2, 1])

with v_col2:
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    scatter = ax2.scatter(df[x_axis], df[y_axis], c=df['Cluster'], cmap='viridis', alpha=0.6, s=15)
    
    # Plotting Centroids
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    x_idx, y_idx = selected_features.index(x_axis), selected_features.index(y_axis)
    ax2.scatter(centers[:, x_idx], centers[:, y_idx], s=100, marker='X', c='red', label='Centroids')
    
    ax2.set_xlabel(x_axis, fontsize=9)
    ax2.set_ylabel(y_axis, fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=7)
    st.pyplot(fig2, use_container_width=True)

# --------------------------------
# Data & Strategy
# --------------------------------

st.subheader("📌 Cluster Profile (Average Spending)")
cluster_profile = df.groupby('Cluster')[selected_features].mean()
st.dataframe(cluster_profile.style.highlight_max(axis=0), use_container_width=True)

st.subheader("💼 Business Strategy Suggestions")
b_cols = st.columns(k_value)
for i in range(k_value):
    with b_cols[i]:
        st.markdown(f"**Cluster {i}**")
        avg = cluster_profile.loc[i].mean()
        if avg < 5000: st.success("Low-value: Promotions")
        elif avg < 12000: st.info("Mid-value: Loyalty")
        else: st.warning("High-value: Priority")

st.markdown("---")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("⬇ Download Results", data=csv, file_name="clustered_data.csv", mime="text/csv")