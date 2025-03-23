import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score

# from google.colab import files
# uploaded = files.upload()

# Load the dataset (Download from Kaggle first)
file_path = "Mall_Customers.csv"
df = pd.read_csv(file_path)

# Print dataset overview
print("-" * 160)
print("\nDataset Preview:\n\n", df.head())
print("-" * 160)
print("\nDataset Info:\n")
df.info()

print("-" * 160)
# Print dataset description
print("\n Dataset Description \n\n", df.describe())

# Selecting relevant features for clustering
df = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

print("-" * 160)
# Print raw data before preprocessing
print("\nRaw Data (Before Preprocessing):\n\n", df.head())

# Handle missing values (if any)
imputer = SimpleImputer(strategy="mean")
df.iloc[:, :] = imputer.fit_transform(df)

# Standardize features (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Convert back to DataFrame for better visualization
df_scaled = pd.DataFrame(X_scaled, columns=['Age', 'Annual_Income', 'Spending_Score'])

print("-" * 160)
# Print data after preprocessing
print("\nPreprocessed Data (After Cleaning & Standardization):\n\n", df_scaled.head())

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Apply Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=5)
agglo_labels = agglo.fit_predict(X_scaled)

# Plot Clustering Results
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis', edgecolor='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label="Centroids")
plt.title("K-Means Clustering")
plt.xlabel("Age")
plt.ylabel("Annual Income")
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='viridis', edgecolor='k', s=50)
plt.title("DBSCAN Clustering")
plt.xlabel("Age")
plt.ylabel("Annual Income")

plt.subplot(1, 3, 3)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=agglo_labels, cmap='viridis', edgecolor='k', s=50)
plt.title("Agglomerative Clustering")
plt.xlabel("Age")
plt.ylabel("Annual Income")

print("-" * 160)
plt.show()

print("-" * 160)
# Testing on 5 test instances
print("\nTesting on 5 Test Instances:")
test_sample_indices = np.random.choice(X_test.shape[0], 5, replace=False)  # Select 5 random test points

test_results = []
for i, idx in enumerate(test_sample_indices):
    test_instance = X_test[idx].reshape(1, -1)
    kmeans_cluster = kmeans.predict(test_instance)[0]
    dbscan_cluster = dbscan.fit_predict(test_instance)[0]
    agglo_cluster = fcluster(linkage_matrix, t=5, criterion='maxclust')[idx]
    
    test_results.append([i+1, kmeans_cluster, dbscan_cluster, agglo_cluster])

test_df = pd.DataFrame(test_results, columns=['Instance', 'K-Means Cluster', 'DBSCAN Cluster', 'Agglomerative Cluster'])
print(test_df.to_string(index=False))

# Compute silhouette scores
silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)

# Check if DBSCAN has at least 2 clusters (excluding noise)
dbscan_cluster_labels = dbscan_labels[dbscan_labels != -1]
if len(set(dbscan_cluster_labels)) > 1:
    silhouette_dbscan = silhouette_score(X_scaled[dbscan_labels != -1], dbscan_cluster_labels)
else:
    silhouette_dbscan = -1  # Invalid silhouette score

silhouette_agglo = silhouette_score(X_scaled, agglo_labels)

print("-" * 160)
# Print silhouette scores
print("\nSilhouette Scores (Higher is Better):\n")
print(f"  K-Means: {silhouette_kmeans:.4f}")
print(f"  DBSCAN: {silhouette_dbscan:.4f}")
print(f"  Agglomerative: {silhouette_agglo:.4f}")

# Determine best algorithm based on silhouette score
best_algorithm = "K-Means" if silhouette_kmeans > max(silhouette_dbscan, silhouette_agglo) else (
    "DBSCAN" if silhouette_dbscan > max(silhouette_kmeans, silhouette_agglo) else "Agglomerative Clustering"
)

# Compute number of clusters found
kmeans_clusters = len(set(kmeans_labels))
dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
agglo_clusters = len(set(agglo_labels))

# Comparison Table
comparison_data = {
    "Algorithm": ["K-Means", "DBSCAN", "Agglomerative Clustering"],
    "No. of Clusters Found": [kmeans_clusters, dbscan_clusters, agglo_clusters],
    "Silhouette Score": [silhouette_kmeans, silhouette_dbscan, silhouette_agglo],
    "Handles Noise?": ["No", "Yes (-1 means noise)", "No"],
    "Cluster Formation": [
        "Centroid-based (Minimizes distance to cluster center)",
        "Density-based (Groups based on dense regions)",
        "Hierarchical-based (Merges closest clusters iteratively)"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("-" * 160)
print("\nComparison Table:\n")
print(comparison_df.to_string(index=False))

print("-" * 160)
# Print conclusion
print("\n Conclusion:\n")
print(f"Based on silhouette scores and cluster quality, the best algorithm for this dataset is: **{best_algorithm}**")

if best_algorithm == "K-Means":
    print("- K-Means provides well-separated, balanced clusters.")
    print("- Best for structured datasets with clear cluster boundaries.")
elif best_algorithm == "DBSCAN":
    print("- DBSCAN is best for detecting noise and handling non-uniform density clusters.")
    print("- Useful for datasets with outliers or irregular cluster shapes.")
elif best_algorithm == "Agglomerative Clustering":
    print("- Agglomerative Clustering performs well when hierarchical relationships exist.")
    print("- Best suited for datasets where merging small clusters makes sense.")

print("\n🔍 If noise is present, DBSCAN might be preferable. Otherwise, K-Means generally works best for this dataset.")
