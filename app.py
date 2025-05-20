import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy
import sklearn
import streamlit as st
np.random.seed(1)
SEED=1


def segmentation_model(X, n_clusters=4, algorithm='MiniBatch KMeans'):
    if algorithm == 'MiniBatch KMeans':
        model = sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters,
                                                init='k-means++', max_iter=100, random_state=SEED)
    elif algorithm == 'Elkan KMeans':
        model = sklearn.cluster.KMeans(n_clusters=n_clusters, init='k-means++',
                                       max_iter=300, random_state=SEED)
    else:
        raise ValueError("Choose one of the two algorithms - mini_batch or elkan")
    
    cluster_labels = model.fit_predict(X)

    return model, cluster_labels

segmentation_goals = {
    "General Customer Profile": ['Age', 'Income', 'Total_Spend', 'Web_Engagement'],
    "Loyalty Segmentation": ['Age', 'Income', 'Customer_Tenure'],
    "Channel Engagement": ['Web_Engagement', 'Catalog_Engagement', 'Store_Engagement'],
    "Promotional Targeting": ['Campaign_Acceptance_Rate', 'Response_Last_Campaign', 'Total_Spend'],
    "Product Preference": ['MntWines_Ratio', 'MntMeatProducts_Ratio', 'Total_Spend'],
    "High-Value Customers": ['Total_Spend', 'Avg_Spend_Per_Purchase', 'Customer_Tenure'],
    "Discount Sensitivity": ['Discount_Effect', 'Total_Spend', 'NumDealsPurchases'],
    "Recent Activity": ['Recency', 'Web_Engagement', 'Total_Purchases'],
    "Purchase Frequency": ['Purchase_Frequency_Monthly', 'Avg_Spend_Per_Purchase', 'Total_Spend'],
    "Customer Lifecycle Stage": ['Customer_Tenure', 'Recency', 'Response_Last_Campaign']
}

goal_des = {
    "General Customer Profile": "Segment customers based on broad traits like age, income, and spend.",
    "Loyalty Segmentation": "Identify long-term vs. new customers based on tenure, income, and age.",
    "Channel Engagement": "See how customers differ by web, catalog, and store usage.",
    "Promotional Targeting": "Find groups most responsive to marketing campaigns.",
    "Product Preference": "Group customers based on what they like to buy the most.",
    "High-Value Customers": "Identify customers with high spending and long tenure.",
    "Discount Sensitivity": "Segment customers based on their responsiveness to discounts.",
    "Recent Activity": "Group customers by their recent engagement and purchase activity.",
    "Purchase Frequency": "Segment customers based on how often they make purchases.",
    "Customer Lifecycle Stage": "Identify customers at different stages of their lifecycle with the company."
}

cluster_feature_options = {
    'Age':'Age', 'Income': 'Income', 'Family Size': 'Family_Size', 'Total Spending': 'Total_Spend', 'Total Number of Purchases': 'Total_Purchases', 'Customer Tenure': 'Customer_Tenure', 'Web Engagement': 'Web_Engagement', 'Average Spending': 'Avg_Spend_Per_Purchase', 'Catalog Engagement': 'Catalog_Engagement', 'Web Engaement' : 'Web_Engagement',
    'Store Engagement': 'Store_Engagement', 'Web Purchase Ratio': 'Web_Purchase_Ratio', 'Discount Effect': 'Discount_Effect', 'Total Campaigns Accepted': 'Total_Campaigns_Accepted', 'Accepted Last Campaign?': 'Response_Last_Campaign', 'Last Purchase' : 'Recency', 'Campaign Acceptance  Rate': 'Campaign_Acceptance_Rate',
    'Most Purchased Category': 'Dominant_Category_Enc', 'Average Spending on Wine': 'MntWines_Ratio', 'Average Spending on Fruits': 'MntFruits_Ratio', 'Average Spending on Meat': 'MntMeatProducts_Ratio', 'Average Spending on Fish': 'MntFishProducts_Ratio',
    'Average Spending on Sweets': 'MntSweetProducts_Ratio', 'Average Spending on Gold': 'MntGoldProds_Ratio', 'Monthly Purchase Frequency': 'Purchase_Frequency_Monthly', 'Age Group': 'Age_Group_Enc', 'Marital Status': 'Marital_Status_Enc', 'Educational Qualification': 'Education_Enc'}

def pca_decomposition(X, labels):
    pca = sklearn.decomposition.PCA(n_components=2, random_state=SEED)
    components = pca.fit_transform(X)
    pca_features = pd.DataFrame(components, columns=['PC1', 'PC2'])
    pca_features['Cluster'] = labels
    return pca_features

def cluster_summary(X, labels):
    clustered = X.copy()
    clustered["Cluster"] = labels
    return clustered.groupby("Cluster").mean().round(2)

def pca_plot(df):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="PC1", y="PC2", hue="Cluster", data=df, palette="Set2", s=80)
    plt.title("Customer Segments (PCA Projection)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    st.pyplot(plt.gcf())

def plot_cluster_bars(df, labels, feature):
    df_clustered = df.copy()
    df_clustered["Cluster"] = labels
    mean_values = df_clustered.groupby("Cluster")[feature].mean().sort_index()

    fig, ax = plt.subplots()
    mean_values.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title(f"Average {feature} by Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel(feature)
    st.pyplot(fig)

def plot_boxplot(df, labels, feature):
    df_clustered = df.copy()
    df_clustered["Cluster"] = labels

    fig, ax = plt.subplots()
    sns.boxplot(data=df_clustered, x="Cluster", y=feature, palette="Set3", ax=ax)
    ax.set_title(f"{feature} Distribution by Cluster")
    st.pyplot(fig)

def plot_dominant_category_pie(df, labels, category_col):
    df_clustered = df.copy()
    df_clustered["Cluster"] = labels

    for cluster in sorted(df_clustered["Cluster"].unique()):
        st.markdown(f"### Cluster {cluster} – Dominant Product Category")
        cluster_data = df_clustered[df_clustered["Cluster"] == cluster]
        value_counts = cluster_data[category_col].value_counts()

        fig, ax = plt.subplots()
        wedges, _, autotexts = ax.pie(value_counts.values, labels=value_counts.index, autopct="%1.1f%%", startangle=90, textprops={'fontsize': 10})

        ax.legend(wedges, value_counts.index, title="Categories",loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10, title_fontsize=11)
        ax.axis('equal')
        st.pyplot(fig)

def plot_stacked_bars(df, labels, columns, title="Product Ratios by Cluster"):
    df_clustered = df.copy()
    df_clustered["Cluster"] = labels
    mean_ratios = df_clustered.groupby("Cluster")[columns].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(mean_ratios))

    for col in columns:
        ax.bar(mean_ratios.index, mean_ratios[col], bottom=bottom, label=col)
        bottom += mean_ratios[col]

    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proportion")
    ax.legend(loc="best")
    st.pyplot(fig)

@st.cache_data
def load_data():
    return pd.read_csv("app_dataset.csv")

st_data = load_data()
st_data_scaled = st_data.copy()

features_to_scale = ['Income', 'Recency', 'Age', 'Family_Size', 'Customer_Tenure','Total_Spend', 'Total_Purchases', 'Avg_Spend_Per_Purchase', 'Web_Engagement', 'Store_Engagement', 'Catalog_Engagement', 'Web_Purchase_Ratio','Discount_Effect', 'Total_Campaigns_Accepted', 'Campaign_Acceptance_Rate','MntWines_Ratio', 'MntFruits_Ratio', 'MntMeatProducts_Ratio', 'MntFishProducts_Ratio','MntSweetProducts_Ratio', 'MntGoldProds_Ratio', 'Purchase_Frequency_Monthly']
st_data_scaler = sklearn.preprocessing.StandardScaler()
st_data_scaled[features_to_scale] = st_data_scaler.fit_transform(st_data_scaled[features_to_scale])



# Sidebar code

st.sidebar.title("Customize")

n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=15, value=5)

algorithm = st.sidebar.radio("Clustering Algorithm", ["MiniBatch KMeans", "Elkan KMeans"])

segmentation_mode = st.sidebar.radio(
    "Choose Feature Selection Mode:",
    ["Use Segmentation Goal", "Select Custom Features"])
if segmentation_mode == "Use Segmentation Goal":
    goal_choice = st.sidebar.selectbox("Segmentation Goal", list(segmentation_goals.keys()))
    st.sidebar.caption(goal_des[goal_choice])
    selected_features = segmentation_goals[goal_choice]

elif segmentation_mode == "Select Custom Features":
    custom_features = st.sidebar.multiselect(
        "Select Features for Clustering",
        options=list(cluster_feature_options.keys())
    )
    selected_features = [cluster_feature_options[feature] for feature in custom_features]
    


# Main code for dashboard

st.title("Customer Segmentation")
st.markdown("Group your customers into meaningful segments")

if not selected_features:
    st.warning("Please select at least one feature to proceed with clustering.")

if selected_features:
    X = st_data_scaled[selected_features]
    model, labels = segmentation_model(X, n_clusters=n_clusters, algorithm=algorithm)
    st_data["Cluster"] = labels

    pca_X = pca_decomposition(st_data, labels)

    with st.expander("Cluster Overview (PCA)", expanded=True):
        st.markdown("The PCA scatter plot projects customers into a 2D space, colored by cluster. This helps visualize how distinct the clusters are based on the selected features.")
        pca_plot(pca_X)

    with st.expander("Cluster Summary Table", expanded=True):
        st.markdown("This table shows the average values of key features for each cluster. It helps identify the dominant traits of each customer segment.")
        summary = cluster_summary(st_data, labels)
        st.dataframe(summary)

    with st.expander("Feature Comparisons by Cluster"):
        st.markdown("The bar and box plots below show how each selected feature varies across clusters. This helps interpret what differentiates the groups.")
        for feature in selected_features:
            st.markdown(f"**{feature}**")
            plot_cluster_bars(st_data, labels, feature)
            plot_boxplot(st_data, labels, feature)

    with st.expander("Distribution of Common Features"):
        st.markdown("Boxplots for income and recency help assess financial value and recent engagement across customer groups.")
        for common_feature in ["Income", "Recency"]:
            if common_feature in st_data.columns:
                st.markdown(f"**{common_feature}**")
                plot_boxplot(st_data, labels, common_feature)

    if "Dominant_Category_Enc" in st_data.columns:
        with st.expander("Product Category Preference"):
            st.markdown("Pie charts show the most preferred product category within each cluster, giving insight into dominant buying behaviors.")
            plot_dominant_category_pie(st_data, labels, "Dominant_Category_Enc")

    product_ratio_cols = [col for col in st_data.columns if col.endswith("_Ratio")]
    if product_ratio_cols:
        with st.expander("Product Spend Composition"):
            st.markdown("Stacked bars show how each cluster allocates spending across product types, offering a breakdown of shopping priorities.")
            plot_stacked_bars(st_data, labels, product_ratio_cols[:5])