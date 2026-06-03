import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Load and clean data
df = pd.read_csv("train.feats.numeric.csv")
df.columns = df.columns.str.strip()

# 2. Drop administrative, date, and ID fields
to_drop = [
    "Form Name", "User Name", "Hospital",
    "id-hushed_internalpatientid", "surgery before or after-Activity date",
    "אבחנה-Diagnosis date",
    "אבחנה-Surgery date1", "אבחנה-Surgery date2", "אבחנה-Surgery date3",
    "אבחנה-Surgery name1", "אבחנה-Surgery name2", "אבחנה-Surgery name3",
    "אבחנה-Surgery sum"
]
df.drop(columns=[c for c in to_drop if c in df.columns], inplace=True)

# 3. Define feature sets
bio_feats = [
    "אבחנה-er",
    "אבחנה-pr",
    "אבחנה-Her2",
    "אבחנה-KI67 protein",
    "אבחנה-Tumor width",
    "אבחנה-Tumor depth",
    "אבחנה-T -Tumor mark (TNM)"
]
total_numeric = df.select_dtypes(include='number').columns.tolist()
other_feats = [c for c in total_numeric if c not in bio_feats]

# 4. PCA + K-Means on biological features
X = df[bio_feats]
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(X_scaled)
kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(X_pca)

# 5. Plot PCA scatter with clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='tab10', s=30, alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', s=100, marker='X', label='Centroids')
plt.title('PCA (Hormone + Morphology) with K-Means Clusters')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

# 6. Non-bio feature variation by cluster
print("\n***** Non-bio Feature Variation by Cluster *****")
for cluster in sorted(df['Cluster'].unique()):
    stds = df[df['Cluster'] == cluster][other_feats].std().sort_values()
    print(f"\nCluster {cluster} — Least-varying non-bio features:")
    print(stds.head(5))
    print(f"Cluster {cluster} — Most-varying non-bio features:")
    print(stds.tail(5))

# 7. Compute cluster centroids for all numeric features
centroids = df.groupby('Cluster')[total_numeric].mean()

# 8. Distinguishing features with values between clusters
high_entries = []
low_entries = []
for i in centroids.index:
    others = centroids.drop(i)
    # uniquely high: difference from next highest
    next_high = others.max()
    diff_high = centroids.loc[i] - next_high
    top_high = diff_high.sort_values(ascending=False).head(5)
    for feat, diff in top_high.items():
        high_entries.append({
            'Cluster': i,
            'Feature': feat,
            'ClusterMean': centroids.loc[i, feat],
            'NextHighestMean': next_high[feat],
            'MeanDiff': diff
        })
    # uniquely low: difference from next lowest
    next_low = others.min()
    diff_low = centroids.loc[i] - next_low
    top_low = diff_low.sort_values().head(5)
    for feat, diff in top_low.items():
        low_entries.append({
            'Cluster': i,
            'Feature': feat,
            'ClusterMean': centroids.loc[i, feat],
            'NextLowestMean': next_low[feat],
            'MeanDiff': diff
        })
# Create DataFrames
high_df = pd.DataFrame(high_entries)
low_df = pd.DataFrame(low_entries)

# 9. Display distinguishing values with feature names
print("\nTop 5 Features Uniquely High by Cluster with Values:")
for cluster in sorted(high_df['Cluster'].unique()):
    print(f"\nCluster {cluster}:")
    subset = high_df[high_df['Cluster'] == cluster]
    for _, row in subset.iterrows():
        print(f"  Feature: {row['Feature']}, ClusterMean = {row['ClusterMean']:.2f}, "
              f"NextHighestMean = {row['NextHighestMean']:.2f}, Diff = {row['MeanDiff']:.2f}")

print("\nTop 5 Features Uniquely Low by Cluster with Values:")
for cluster in sorted(low_df['Cluster'].unique()):
    print(f"\nCluster {cluster}:")
    subset = low_df[low_df['Cluster'] == cluster]
    for _, row in subset.iterrows():
        print(f"  Feature: {row['Feature']}, ClusterMean = {row['ClusterMean']:.2f}, "
              f"NextLowestMean = {row['NextLowestMean']:.2f}, Diff = {row['MeanDiff']:.2f}")
