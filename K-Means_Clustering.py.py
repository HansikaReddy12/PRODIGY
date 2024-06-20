import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, features):
    scaler = StandardScaler()
    X = df[features]
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def find_optimal_k(X_scaled, max_k=10):
    inertia = []
    K = range(1, max_k + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    
    # Plotting the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K, inertia, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.show()

def perform_kmeans(X_scaled, optimal_k):
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    return clusters

def evaluate_clusters(X_scaled, clusters):
    silhouette_avg = silhouette_score(X_scaled, clusters)
    print(f'Silhouette Score: {silhouette_avg}')

def visualize_clusters(df, clusters, X_scaled):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    df['Cluster'] = clusters
    
    plt.figure(figsize=(10, 6))
    for cluster in range(np.max(clusters) + 1):
        subset = df[df['Cluster'] == cluster]
        plt.scatter(subset['PCA1'], subset['PCA2'], label=f'Cluster {cluster}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Customer Segments Visualization')
    plt.legend()
    plt.show()

# File path to CSV
file_path = "/Users/ashis/Desktop/spyder/Mall_Customers.csv"

# Features
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']  # Assuming these are the columns for clustering

# Optimal number of clusters
optimal_k = 3

# Main function
def main():
    df = load_data(file_path)
    X_scaled = preprocess_data(df, features)
    
    find_optimal_k(X_scaled, max_k=10)
    
    clusters = perform_kmeans(X_scaled, optimal_k)
    
    evaluate_clusters(X_scaled, clusters)
    
    visualize_clusters(df, clusters, X_scaled)

if __name__ == "__main__":
    main()
