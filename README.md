### 1. Importing Libraries

```python
import numpy as np
```
- **Purpose:** Imports the NumPy library and assigns it the alias `np`. NumPy is used for efficient numerical operations, especially with arrays.

```python
import pandas as pd
```
- **Purpose:** Imports the pandas library (alias `pd`), which is widely used for data manipulation and analysis (e.g., loading CSV files into DataFrames).

```python
import matplotlib.pyplot as plt
```
- **Purpose:** Imports the `pyplot` module from Matplotlib (alias `plt`) for data visualization and plotting.

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
```
- **Purpose:** Imports three clustering algorithms from scikit-learn:
  - **KMeans:** A centroid-based clustering algorithm.
  - **DBSCAN:** A density-based clustering algorithm.
  - **AgglomerativeClustering:** A hierarchical clustering method.

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
```
- **Purpose:** 
  - **StandardScaler:** Standardizes features (zero mean and unit variance), which is crucial for many machine learning algorithms.
  - **LabelEncoder:** Encodes categorical labels as numeric values (although in this code, it is imported but not used).

```python
from sklearn.impute import SimpleImputer
```
- **Purpose:** Imports the `SimpleImputer` class, used to handle missing values by imputing them with a specified strategy (here, using the mean).

```python
from sklearn.metrics import silhouette_score
```
- **Purpose:** Imports the `silhouette_score` function to evaluate the quality of clustering by measuring how similar an object is to its own cluster compared to other clusters.

```python
from google.colab import files
```
- **Purpose:** Imports the `files` module from Google Colab, which allows users to upload files from their local system when running in a Colab notebook.

---

### 2. Uploading and Loading the Dataset

```python
uploaded = files.upload()
```
- **Purpose:** Opens a file upload dialog in Google Colab to let the user upload the dataset file(s).

```python
# Load the dataset (Download from Kaggle first)
file_path = "Mall_Customers.csv"
df = pd.read_csv(file_path)
```
- **Purpose:** 
  - **file_path:** Specifies the file name of the dataset.
  - **pd.read_csv(file_path):** Reads the CSV file into a pandas DataFrame named `df`.

---

### 3. Exploring the Dataset

```python
# Print dataset overview
print("\nDataset Preview:\n", df.head())
print("\nDataset Info:\n")
df.info()
```
- **Purpose:**
  - **df.head():** Displays the first few rows of the dataset to give a quick preview.
  - **df.info():** Provides detailed information about the DataFrame (e.g., column types, non-null counts) to help understand the data structure.

---

### 4. Feature Selection

```python
# Selecting relevant features for clustering
df = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
```
- **Purpose:** Filters the DataFrame to keep only the columns "Age", "Annual Income (k$)", and "Spending Score (1-100)", which are considered relevant for clustering analysis.

```python
# Print raw data before preprocessing
print("\nRaw Data (Before Preprocessing):\n", df.head())
```
- **Purpose:** Prints the first few rows of the selected data to inspect the raw input before further processing.

---

### 5. Data Preprocessing

#### Handling Missing Values

```python
# Handle missing values (if any)
imputer = SimpleImputer(strategy="mean")
df.iloc[:, :] = imputer.fit_transform(df)
```
- **Purpose:**
  - **SimpleImputer(strategy="mean"):** Creates an imputer that replaces missing values with the mean of each column.
  - **imputer.fit_transform(df):** Fits the imputer on the DataFrame and transforms the data, ensuring there are no missing values.

#### Standardizing the Features

```python
# Standardize features (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
```
- **Purpose:**
  - **StandardScaler():** Instantiates a scaler that standardizes the features.
  - **scaler.fit_transform(df):** Fits the scaler on the data and then transforms it so that each feature has a mean of 0 and a standard deviation of 1. The result is stored in `X_scaled`.

```python
# Convert back to DataFrame for better visualization
df_scaled = pd.DataFrame(X_scaled, columns=['Age', 'Annual_Income', 'Spending_Score'])
```
- **Purpose:** Converts the scaled NumPy array back into a pandas DataFrame with clearer column names for visualization and further analysis.

```python
# Print data after preprocessing
print("\nPreprocessed Data (After Cleaning & Standardization):\n", df_scaled.head())
```
- **Purpose:** Prints the first few rows of the preprocessed (cleaned and standardized) data.

---

### 6. Applying Clustering Algorithms

#### K-Means Clustering

```python
# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
```
- **Purpose:**
  - **KMeans(n_clusters=5, random_state=42, n_init=10):** Creates a K-Means clustering object with 5 clusters, a fixed random seed for reproducibility, and 10 different initializations to choose the best one.
  - **kmeans.fit_predict(X_scaled):** Fits the K-Means model on the standardized data and assigns a cluster label to each data point.

#### DBSCAN Clustering

```python
# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
```
- **Purpose:**
  - **DBSCAN(eps=0.5, min_samples=5):** Instantiates the DBSCAN algorithm with an epsilon of 0.5 (radius of neighborhood) and a minimum of 5 samples required to form a dense region.
  - **dbscan.fit_predict(X_scaled):** Fits DBSCAN on the data and predicts cluster labels (with `-1` indicating noise).

#### Agglomerative Clustering

```python
# Apply Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=5)
agglo_labels = agglo.fit_predict(X_scaled)
```
- **Purpose:**
  - **AgglomerativeClustering(n_clusters=5):** Creates a hierarchical clustering object that will merge clusters iteratively until 5 clusters remain.
  - **agglo.fit_predict(X_scaled):** Fits the model and assigns a cluster label to each point.

---

### 7. Evaluating Cluster Quality

#### Silhouette Score Calculation

```python
# Compute silhouette scores
silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
```
- **Purpose:** Calculates the silhouette score for the K-Means clustering, which measures how well each data point fits into its cluster compared to other clusters.

```python
# Check if DBSCAN has at least 2 clusters (excluding noise)
dbscan_cluster_labels = dbscan_labels[dbscan_labels != -1]
if len(set(dbscan_cluster_labels)) > 1:
    silhouette_dbscan = silhouette_score(X_scaled[dbscan_labels != -1], dbscan_cluster_labels)
else:
    silhouette_dbscan = -1  # Invalid silhouette score
```
- **Purpose:**
  - **dbscan_labels[dbscan_labels != -1]:** Filters out noise points (labeled `-1`) from DBSCAN results.
  - **if len(set(...)) > 1:** Checks if there are at least two clusters (without noise) so that a silhouette score can be computed.
  - **silhouette_score(...):** Computes the silhouette score on the non-noise points; if not, assigns `-1` to indicate an invalid score.

```python
silhouette_agglo = silhouette_score(X_scaled, agglo_labels)
```
- **Purpose:** Computes the silhouette score for the Agglomerative Clustering result.

#### Printing the Scores

```python
# Print silhouette scores
print("\nSilhouette Scores (Higher is Better):")
print(f"  K-Means: {silhouette_kmeans:.4f}")
print(f"  DBSCAN: {silhouette_dbscan:.4f}")
print(f"  Agglomerative: {silhouette_agglo:.4f}")
```
- **Purpose:** Prints the computed silhouette scores for each algorithm, formatted to four decimal places.

---

### 8. Determining the Best Clustering Algorithm

```python
# Determine best algorithm based on silhouette score
best_algorithm = "K-Means" if silhouette_kmeans > max(silhouette_dbscan, silhouette_agglo) else (
    "DBSCAN" if silhouette_dbscan > max(silhouette_kmeans, silhouette_agglo) else "Agglomerative Clustering"
)
```
- **Purpose:** Compares the silhouette scores of the three clustering methods and assigns the name of the algorithm with the highest score to the variable `best_algorithm`.

```python
# Compute number of clusters found
kmeans_clusters = len(set(kmeans_labels))
dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
agglo_clusters = len(set(agglo_labels))
```
- **Purpose:**
  - **kmeans_clusters:** Counts the unique clusters found by K-Means.
  - **dbscan_clusters:** Counts the unique clusters from DBSCAN while subtracting one if the noise label (`-1`) is present.
  - **agglo_clusters:** Counts the clusters formed by Agglomerative Clustering.

---

### 9. Creating a Comparison Table

```python
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
```
- **Purpose:** Constructs a dictionary that summarizes each algorithm’s performance, including:
  - Name of the algorithm.
  - Number of clusters found.
  - Their silhouette score.
  - Whether they handle noise.
  - A brief note on how they form clusters.

```python
comparison_df = pd.DataFrame(comparison_data)
print("\nComparison Table:\n")
print(comparison_df.to_string(index=False))
```
- **Purpose:**
  - **pd.DataFrame(comparison_data):** Converts the dictionary into a pandas DataFrame.
  - **comparison_df.to_string(index=False):** Prints the table in a clean format without showing row indices.

---

### 10. Visualizing the Clustering Results

#### Setting Up the Plot

```python
# Plot Clustering Results
plt.figure(figsize=(18, 5))
```
- **Purpose:** Creates a new figure for plotting with a specified size of 18 inches in width and 5 inches in height.

#### K-Means Clustering Plot

```python
plt.subplot(1, 3, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis', edgecolor='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label="Centroids")
plt.title("K-Means Clustering")
plt.xlabel("Age")
plt.ylabel("Annual Income")
plt.legend()
```
- **Purpose:**
  - **plt.subplot(1, 3, 1):** Divides the figure into a 1×3 grid and selects the first subplot.
  - **plt.scatter(...):** Plots the data points from the first two standardized features, coloring them by their K-Means cluster label.
  - **Second plt.scatter:** Overlays the cluster centroids (in red with an "x" marker) on the plot.
  - **Title, labels, legend:** Adds context to the plot.

#### DBSCAN Clustering Plot

```python
plt.subplot(1, 3, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='viridis', edgecolor='k', s=50)
plt.title("DBSCAN Clustering")
plt.xlabel("Age")
plt.ylabel("Annual Income")
```
- **Purpose:**
  - **plt.subplot(1, 3, 2):** Selects the second subplot in the grid.
  - **plt.scatter(...):** Plots the data points colored by their DBSCAN labels. Points labeled `-1` (noise) will appear as a separate color.
  - **Title and labels:** Provides context for the DBSCAN plot.

#### Agglomerative Clustering Plot

```python
plt.subplot(1, 3, 3)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=agglo_labels, cmap='viridis', edgecolor='k', s=50)
plt.title("Agglomerative Clustering")
plt.xlabel("Age")
plt.ylabel("Annual Income")
```
- **Purpose:**
  - **plt.subplot(1, 3, 3):** Selects the third subplot.
  - **plt.scatter(...):** Plots the data points colored by the cluster labels produced by Agglomerative Clustering.
  - **Title and labels:** Adds context to the plot.

```python
plt.show()
```
- **Purpose:** Displays all three subplots in the figure.

---

### 11. Printing the Conclusion

```python
# Print conclusion
print("\n### Conclusion ###")
print(f"Based on silhouette scores and cluster quality, the best algorithm for this dataset is: **{best_algorithm}**")
```
- **Purpose:** Prints a summary conclusion stating which clustering algorithm performed best based on the computed silhouette scores.

```python
if best_algorithm == "K-Means":
    print("- K-Means provides well-separated, balanced clusters.")
    print("- Best for structured datasets with clear cluster boundaries.")
elif best_algorithm == "DBSCAN":
    print("- DBSCAN is best for detecting noise and handling non-uniform density clusters.")
    print("- Useful for datasets with outliers or irregular cluster shapes.")
elif best_algorithm == "Agglomerative Clustering":
    print("- Agglomerative Clustering performs well when hierarchical relationships exist.")
    print("- Best suited for datasets where merging small clusters makes sense.")
```
- **Purpose:** Based on which algorithm was chosen as best, prints additional tailored insights explaining why that algorithm is advantageous for the dataset.

```python
print("\n🔍 If noise is present, DBSCAN might be preferable. Otherwise, K-Means generally works best for this dataset.")
```
- **Purpose:** Offers a final recommendation by highlighting that while DBSCAN is excellent at handling noise, K-Means is often preferred when the data is clean and has well-separated clusters.

![](https://github-production-user-asset-6210df.s3.amazonaws.com/126332769/444201297-7da704cc-9ca1-4f0f-ba8e-19b92d03b3aa.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250515%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250515T161010Z&X-Amz-Expires=300&X-Amz-Signature=a66ef6422b4019cd03d3ad7188bef86e93b0928591c76179de404e3801061166&X-Amz-SignedHeaders=host)

---