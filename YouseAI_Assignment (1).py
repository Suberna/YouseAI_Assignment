#!/usr/bin/env python
# coding: utf-8

# # Importing the Dependencies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# # Loading the dataset

# In[2]:


#Load the dataset
dataset = pd.read_csv("Dataset.csv")


# # Data Preprocessing

# In[6]:


#First 5 rows in the dataframe
dataset.head()


# In[7]:


#Finding the number of rows and columns
dataset.shape


# In[8]:


#Getting information about the dataset
dataset.info()


# In[9]:


#Checking for missing values
dataset.isnull().sum()


# In[10]:


# 1. Handling missing values
# 1_ Drop non-nemeric columns like "CUST_ID"
if "CUST_ID" in dataset.columns:
  dataset.drop(columns = ["CUST_ID"], inplace = True)

# 2) Ensure all columns are numeric before filling missing values
#Fill missing values with the median of each numeric column
numeric_columns = dataset.select_dtypes(include = ['float64','int64']).columns
dataset[numeric_columns] = dataset[numeric_columns].fillna(dataset[numeric_columns].median())

# 2. Feature Engineering
# 1) Average transaction amount per month
# Assuming 'TENURE' is in months, calculate average transaction amount
dataset['AVG_TRANSACTION_AMOUNT'] = dataset['PURCHASES'] / dataset['TENURE']

# 2) Total credit used vs. credit limit
dataset['TOTAL_CREDIT_USED'] = dataset['BALANCE'] + dataset['CASH_ADVANCE']  # total credit used is the balance + cash advance
dataset['CREDIT_USED_RATIO'] = dataset['TOTAL_CREDIT_USED'] / dataset['CREDIT_LIMIT']  # ratio of credit used to credit limit

# 3) Frequency of transactions
# Calculate frequency of transactions as the total transactions made divided by the tenure
dataset['TRANSACTION_FREQUENCY'] = dataset['PURCHASES_TRX'] / dataset['TENURE']

# 3. Feature Scaling: Use StandardScaler to standardize numerical features
scaler = StandardScaler()
scaler_data = scaler.fit_transform(dataset[numeric_columns])

# Convert the scaled data back to DataFrame for later use
scaler_data_df = pd.DataFrame(scaler_data, columns = numeric_columns)

# Displaying the
print('Dataset with new feature')
print(dataset.head())
print("\nScaled Dataset")
print(scaler_data_df.head())


# # Exploratory Data Analysis (EDA)
# 
# 

# In[11]:


# setting plots style
sns.set(style = "whitegrid")

# 1. Histograms for distribution of transaction amounts
plt.figure(figsize=(12, 6))
sns.histplot(dataset['AVG_TRANSACTION_AMOUNT'], bins=30, kde=True)
plt.title('Distribution of Average Transaction Amount')
plt.xlabel('Average Transaction Amount')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# In[12]:


# 2. Box plots to identify outliers in credit usage
plt.figure(figsize=(12, 6))
sns.boxplot(x=dataset['CREDIT_USED_RATIO'])
plt.title('Box Plot of Credit Usage')
plt.xlabel('Credit Usage Ratio')
plt.grid()
plt.show()


# In[11]:


# 3. Heatmap to visualize correlations between features
plt.figure(figsize=(14, 10))
correlation_matrix = dataset.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()


# In[13]:


# 4. Analyze spending patterns and transaction frequencies
# Create a summary DataFrame for spending patterns
spending_summary = dataset[['CREDIT_LIMIT', 'BALANCE', 'PAYMENTS', 'AVG_TRANSACTION_AMOUNT', 'TRANSACTION_FREQUENCY']].describe()
print("Spending Summary:\n", spending_summary)


# In[14]:


# 5. Segment analysis based on transaction frequency
plt.figure(figsize=(12, 6))
sns.countplot(x='TRANSACTION_FREQUENCY', data=dataset, palette='pastel')
plt.title('Transaction Frequency Count')
plt.xlabel('Transaction Frequency')
plt.ylabel('Count of Customers')
plt.grid()
plt.show()


# # Clustering

# In[15]:


# 1. K-Means Clustering
# Use elbow method to find the optimal number of clusters
inertia = []
silhouette_scores_kmeans = []
K_range = range(2, 11)  # Evaluating for K from 2 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaler_data_df)
    inertia.append(kmeans.inertia_)
    silhouette_scores_kmeans.append(silhouette_score(scaler_data_df, kmeans.labels_))

# Plotting the Elbow Method
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')

# Plotting Silhouette Scores for K-Means
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores_kmeans, marker='o', color='orange')
plt.title('Silhouette Scores for K-Means')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()


# In[17]:


# 2. DBSCAN Clustering
# Experiment with different values of epsilon and min_samples
epsilon_values = np.arange(0.1, 1.5, 0.1)
min_samples = [2, 3, 5]
silhouette_scores_dbscan = {}

for eps in epsilon_values:
    for min_samples_val in min_samples:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples_val)
        dbscan_labels = dbscan.fit_predict(scaler_data_df)
        if len(set(dbscan_labels)) > 1:  # Only compute score if there are at least 2 clusters
            silhouette_avg = silhouette_score(scaler_data_df, dbscan_labels)
            silhouette_scores_dbscan[(eps, min_samples_val)] = silhouette_avg
        else:
            silhouette_scores_dbscan[(eps, min_samples_val)] = -1  # Invalid silhouette score

# Display DBSCAN results
best_dbscan_params = max(silhouette_scores_dbscan, key=silhouette_scores_dbscan.get)
print(f"Best DBSCAN parameters: epsilon = {best_dbscan_params[0]}, min_samples = {best_dbscan_params[1]}")
print(f"Best silhouette score for DBSCAN: {silhouette_scores_dbscan[best_dbscan_params]}")


# In[18]:


# 3. Visualizing the clusters for the optimal K-Means
optimal_k = silhouette_scores_kmeans.index(max(silhouette_scores_kmeans)) + 2  # +2 because range starts from 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(scaler_data_df)

# Visualizing K-Means Clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x=scaler_data_df.iloc[:, 0], y=scaler_data_df.iloc[:, 1], hue=kmeans_labels, palette='viridis', s=100)
plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# # Cluster Profiling

# In[19]:


# Add the cluster labels to your dataset
dataset['Cluster'] = kmeans_labels


# In[21]:


# 1. Cluster Descriptions
cluster_profiles = dataset.groupby('Cluster').agg({
    'AVG_TRANSACTION_AMOUNT': 'mean',
    'TOTAL_CREDIT_USED': 'sum',
    'TRANSACTION_FREQUENCY': 'mean',
    'BALANCE': 'mean',
    'CREDIT_USED_RATIO': 'mean',
    'PURCHASES': 'sum',
    'PAYMENTS': 'sum'
}).reset_index()

cluster_profiles.columns = ['Cluster',
                            'Average Transaction Amount',
                            'Total Spending',
                            'Average Transaction Frequency',
                            'Average Balance',
                            'Average Credit Usage',
                            'Total Purchases',
                            'Total Payments']

print("Cluster Profiles:")
print(cluster_profiles)


# In[22]:


# 2. Proposed Marketing Strategies
def marketing_strategy(row):
    if row['Total Spending'] > 10000:
        return "Premium rewards and exclusive offers."
    elif row['Total Spending'] > 5000:
        return "Targeted campaigns with personalized offers."
    elif row['Average Transaction Frequency'] < 1:
        return "Incentives for increased usage and cashback offers."
    else:
        return "Regular promotions and retention strategies."

cluster_profiles['Marketing Strategy'] = cluster_profiles.apply(marketing_strategy, axis=1)

# Displaying the cluster profiles with marketing strategies
print("\nCluster Profiles with Marketing Strategies:")
print(cluster_profiles)


# In[23]:


#Define unique marketing strategies for each cluster
def unique_marketing_strategy(row):
    if row['Cluster'] == 0:  # High spenders
        return "Offer premium rewards and exclusive offers to enhance loyalty."
    elif row['Cluster'] == 1:  # Medium spenders
        return "Provide targeted campaigns with personalized offers to encourage higher spending."
    elif row['Cluster'] == 2:  # Low spenders with low frequency
        return "Implement incentives for increased usage, such as cashback offers."
    elif row['Cluster'] == 3:  # Regular users
        return "Introduce regular promotions to maintain engagement and retention."
    elif row['Cluster'] == 4:  # Infrequent users
        return "Encourage re-engagement with special offers for their next purchase."
    else:
        return "General marketing strategies aimed at increasing brand awareness."

# Apply the function to the cluster_profiles DataFrame
cluster_profiles['Unique Marketing Strategy'] = cluster_profiles.apply(unique_marketing_strategy, axis=1)

# Displaying the cluster profiles with unique marketing strategies
print("\nCluster Profiles with Unique Marketing Strategies:")
print(cluster_profiles[['Cluster', 'Unique Marketing Strategy']])


# Principal Component Analysis (PCA)

# In[24]:


# Step 1: Apply PCA
pca = PCA(n_components=2)  # Reducing to 2 dimensions for visualization
pca_data = pca.fit_transform(scaler_data_df)  # Assuming scaler_data_df is your scaled dataset

# Create a DataFrame with PCA components
pca_df = pd.DataFrame(data=pca_data, columns=['PCA1', 'PCA2'])

# Step 2: Clustering on PCA-reduced data
# K-Means Clustering
kmeans_pca = KMeans(n_clusters=3, random_state=42)  # Adjust number of clusters as needed
pca_df['KMeans_Cluster'] = kmeans_pca.fit_predict(pca_df)

# DBSCAN Clustering
dbscan_pca = DBSCAN(eps=0.5, min_samples=5)  # Adjust parameters as needed
pca_df['DBSCAN_Cluster'] = dbscan_pca.fit_predict(pca_df)

# Step 3: Visualization of K-Means Clusters
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='KMeans_Cluster', palette='viridis', s=100)
plt.title('K-Means Clustering (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Step 4: Visualization of DBSCAN Clusters
plt.subplot(1, 2, 2)
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='DBSCAN_Cluster', palette='viridis', s=100)
plt.title('DBSCAN Clustering (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()

# Step 5: Evaluate Clustering Performance
# Silhouette Score for K-Means
kmeans_silhouette = silhouette_score(pca_df[['PCA1', 'PCA2']], pca_df['KMeans_Cluster'])
print(f"K-Means Silhouette Score (PCA): {kmeans_silhouette}")

# Silhouette Score for DBSCAN
# Since DBSCAN can produce noise points, filter them out
if len(set(pca_df['DBSCAN_Cluster'])) > 1:  # Ensure at least 2 clusters for silhouette score
    dbscan_silhouette = silhouette_score(pca_df[['PCA1', 'PCA2']], pca_df['DBSCAN_Cluster'])
    print(f"DBSCAN Silhouette Score (PCA): {dbscan_silhouette}")
else:
    print("DBSCAN did not form sufficient clusters for silhouette score evaluation.")


# In[ ]:




