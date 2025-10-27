import pandas as pd
import datetime as dt
import time


import numpy as np  
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


data=pd.read_csv("D:/Amazon Music Clustering/single_genre_artists.csv")
#finding the null values and duplicates
data.isnull().sum()
data.duplicated().sum()
data['popularity_songs'].value_counts()
# data['duration_ms']= pd.to_datetime(data['duration_ms'])
# data['duration_ms'].dt.time
data['followers'] = data['followers'].astype(int)
data['release_date'] = pd.to_datetime(data['release_date'], format='mixed')
data['year'] = data['release_date'].dt.year
data['year'] = data['year'].astype(int)
data=data.drop(columns=['release_date'])


data.drop(columns=['id_songs','id_artists','name_song','name_artists'],inplace=True)
data['key'].value_counts()
data['explicit'].value_counts()

# Apply log only to duration_ms (since others don't need it)
data['duration_ms'] = np.log1p(data['duration_ms'])

#encoding the categorical columns
data['time_signature'] = data['time_signature'].astype('category')
data['time_signature'] = data['time_signature'].cat.codes
data['key_sin'] = np.sin(2 * np.pi * data['key']/12)
data['key_cos'] = np.cos(2 * np.pi * data['key']/12)
data.drop('key', axis=1, inplace=True)
data['explicit']  =LabelEncoder().fit_transform(data['explicit'])
data['mode'] = LabelEncoder().fit_transform(data['mode'])
data = data.drop(columns=['genres'])

categorical_features= ['time_signature','key_sin','key_cos']
cat_data = pd.DataFrame(data[categorical_features])

# finding the correlation
data.corr()
plt.figure(figsize=(25,25))
plt.title("Correlation between different features")
sns.heatmap(data.corr(),annot=True,cmap='coolwarm')
plt.show()

# finding the outliers
plt.suptitle("Boxplot for different features")
for i in range(1,10):
    plt.subplot(3,3,i)
    sns.boxplot(y=data[data.columns[i]],color='green',orient='v')
plt.show()


#Scaling the data

numeric_features = ['duration_ms', 'danceability', 'energy', 
                    'speechiness', 'acousticness', 'instrumentalness', 
                    'valence', 'tempo']
data_1= data[numeric_features]
data_copy_1 = data_1.copy()
# # Standardizing the numerical features
scaler = MinMaxScaler()
scaled_data_1 = scaler.fit_transform(data_copy_1)

data = pd.concat([pd.DataFrame(scaled_data_1),cat_data.reset_index(drop=True)],axis=1)

# #Applying PCA
pca = PCA(n_components=4)
num_feature= scaler.fit_transform(data_copy_1)
scaled_data = pca.fit_transform(scaled_data_1) 
explained_variance = pca.explained_variance_ratio_
# print("Explained variance by each component:", explained_variance)

loadings = pca.components_
loading_matrix = pd.DataFrame(loadings.T, columns=[f'PC{i+1}' for i in range(len(loadings))], index=data_copy_1.columns)

# print("PCA Loadings:\n", loading_matrix)
pc1_loadings = loading_matrix['PC1'].sort_values(ascending=False)
loading_df = pd.DataFrame({'Feature': data_copy_1.columns, 'PC1 Loading': pc1_loadings.values})
print("Features sorted by PC1 loadings:\n", loading_df)

# Visualizing the loadings for PC1
plt.figure(figsize=(10,6))
sns.barplot(x='Feature', y='PC1 Loading', data=loading_df)
plt.title('Feature Loadings for PC1')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting explained variance
plt.figure(figsize=(8,6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center', label='Individual explained variance')
plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', label='Cumulative explained variance')
plt.xlabel('Principal Component Index')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance')
plt.legend(loc='best')
plt.show()

# clustering the data using Kmeans

sse=[]
# elbow method to find the optimal number of clusters
k_range=range(1,8)
for k in k_range:
    kmeans=KMeans(n_clusters=k,random_state=100,max_iter=1900)
    kmeans.fit_predict(scaled_data)
    sse.append(kmeans.inertia_)
    clusters = kmeans.labels_
plt.plot(k_range,sse)
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.title("Elbow Method")
plt.show()


#evaluating the clusters
score=silhouette_score(scaled_data,clusters)
print(f'Silhouette Score: {score:.2f}')
index=davies_bouldin_score(scaled_data,clusters)
print(f'Davies-Bouldin Index: {index:.2f}')
#visualizing the clusters
data_copy_1['clusters']=clusters
print(data_copy_1['clusters'].value_counts())
centroids=kmeans.cluster_centers_
# print(centroids)
plt.scatter(scaled_data[:,0],scaled_data[:,1],c=clusters,s=50,cmap='viridis',label ='clusters')
plt.scatter(centroids[:,0],centroids[:,1],c='red',s=200,alpha=0.75,label='Centroids')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("KMeans Clustering")
plt.legend()
plt.show()




# # Group by cluster to get mean values per feature
cluster_mean = data_copy_1.groupby('clusters').mean()

cluster_mean = cluster_mean.reset_index('clusters')
print(cluster_mean)
# # Add human-readable labels for each cluster
cluster_labels = {
    0: "🎻 Old Calm Acoustic",
    1: "🎉 Modern Energetic Pop",
    2: "🎤 Rap / Spoken Style",
    3: "💎 Popular Balanced Tracks",
    4: "🔥 High Energy Dance Hits",
    5: "🌙 Mellow Evening Tunes",
    6: "🎸 Rock / Instrumental Focus",
}


# Map the labels
cluster_mean['Cluster_Label'] = cluster_mean['clusters'].map(cluster_labels)

# --- Melt the data for visualization ---
cluster_melt = cluster_mean.melt(
    id_vars=['clusters', 'Cluster_Label'],
    value_vars=data_copy_1.columns.difference(['clusters']),
    var_name='Feature',
    value_name='Value')
# --- 4️⃣ Draw the line plot ---
plt.figure(figsize=(12, 6))
sns.lineplot(x='Feature', y='Value', hue='Cluster_Label',
             data=cluster_melt, marker='o')

plt.title("🎧 Cluster Feature Comparison (with Labels)", fontsize=14)
plt.xlabel("Audio Features")
plt.ylabel("Average Value")
plt.xticks(rotation=30)
plt.legend(title="Cluster Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Heatmap visualization
plt.figure(figsize=(10,6))
sns.heatmap(
    cluster_mean.set_index('Cluster_Label')[data_copy_1.columns.difference(['clusters'])],
    cmap="YlGnBu", annot=True, fmt=".2f"
)
plt.title("🎨 Cluster Feature Heatmap", fontsize=14)
plt.xlabel("Features")
plt.ylabel("Cluster Type")
plt.show()

top_by_popularity = data_copy_1.loc[
    data_copy_1.groupby('clusters')['popularity_songs'].idxmax()
]
print("\nTop tracks by popularity:")
print(top_by_popularity)




# Visualize the cluster centers
plt.figure(figsize=(12, 8))
for i in range(1, len(cluster_mean)):
    plt.bar(range(len(cluster_mean.columns)-1), cluster_mean.iloc[i, :-1], label=f'Cluster {i}', alpha=0.7)
plt.title('Cluster Centers for Each Feature')
plt.xlabel('Features')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#Adding cluster labels to original data
data_copy_1['clusters']=clusters
data['clusters']=clusters
data.to_csv("D:/Amazon Music Clustering/clustered_music_data.csv",index=False)




