Amazon strives to be Earth’s most customer-centric company, Earth’s best employer, and Earth’s safest place to work. 
Customer reviews, 1-Click shopping, personalized recommendations, Prime, Music, Fulfillment by Amazon, AWS, Kindle Direct Publishing, Kindle, Career Choice, Fire tablets, Fire TV, Amazon Echo, Alexa, Just Walk Out technology, Amazon Studios, and The Climate Pledge are some of the things pioneered by Amazon.

Problem Statement:

  With millions of songs available on platforms like Amazon, manually categorizing tracks into genres is impractical. 
  The goal of this project is to automatically group similar songs based on their audio characteristics using clustering techniques. 
  By analyzing patterns in features such as tempo, energy, danceability, and more, learners will develop a model that organizes songs into meaningful clusters, potentially representing different musical genres or moods—without any prior labels.

Dataset features expalanation:
 
| **Feature Name**     | **Description**                                                                |
| -------------------- | ------------------------------------------------------------------------------ |
| **duration_ms**      | Song length in milliseconds (e.g., 12.2 ≈ 2–3 minutes).                        |
| **danceability**     | Measures how suitable a track is for dancing (scale: 0–1).                     |
| **energy**           | Describes the intensity or activity level of a song (higher = more energetic). |
| **loudness**         | Overall average loudness in decibels (lower values = quieter).                 |
| **speechiness**      | Detects the presence of spoken words (higher = more like rap or speech).       |
| **acousticness**     | Likelihood the track is acoustic (1.0 = fully acoustic, 0.0 = electronic).     |
| **instrumentalness** | Probability that a track contains no vocals (1.0 = instrumental only).         |
| **valence**          | Describes the musical positivity or mood (1.0 = happy, 0.0 = sad).             |
| **tempo**            | The speed of the track in beats per minute (BPM).                              |
| **popularity_songs** | The average popularity score of the song (based on user/stream data).          |
| **followers**        | Total number of followers of the artist(s) associated with the track.          |
| **year**             | Release year of the track.                                                     |
| **explicit**         | Indicates explicit content (1 = explicit, 0 = non-explicit).                   |
| **mode**             | Musical mode: 1 = major (happy tone), 0 = minor (sad tone).                    |
| **genres**           | Encoded numerical label for the genre (e.g., Pop=3, Rock=4, etc.).             |


EDA and Data Visualizations:
--->Finding the missing values and duplicates in the dataset.

--->Changing the datatype of Followers and year.

--->Removing the unique features from the dataset such as name and ID of the songs and Artists name and Artists Id.

--->Finding the outliers:

    plt.suptitle("Boxplot for different features")
    for i in range(1,10):
        plt.subplot(3,3,i)
        sns.boxplot(y=data[data.columns[i]],color='green',orient='v')
    plt.show()
    
<img width="640" height="480" alt="boxplot" src="https://github.com/user-attachments/assets/7148559a-b0f0-4105-9a34-53d395a30af2" />

--->Apply log only to duration ms since it has maxixum values 

--->loudness and speechiness can be minimized during the scaling procedure.

--->Encoding the categorical features:

  --->Explicit and mode columns are encoded by ** Labelencoder** method.
  
  --->time_signature is encoded by **category** method.
  
  --->key feature is encoded by **sin and cos cyclical** method.
  
  --->Categorical features are segregated before scaling procedure and only numerical features were scaled by **Standard Scaler ** method.

K-Means Clustering Technique:

  The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares. This algorithm requires the number of clusters to be specified. It scales well to large numbers of samples and has been used across a large range of application areas in many different fields.

  The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion:
  
  Inertia is not a normalized metric: we just know that lower values are better and zero is optimal. But in very high-dimensional spaces, Euclidean distances tend to become inflated (this is an instance of the so-called “curse of dimensionality”). Running a dimensionality reduction algorithm such as Principal component analysis (PCA) prior to k-means clustering can alleviate this problem and speed up the computations.
    
    #Applying PCA
    pca = PCA(n_components=4)
    scaled_data = pca.fit_transform(scaled_data_1) 
    explained_variance = pca.explained_variance_ratio_
    print("Explained variance by each component:", explained_variance)
    
    loadings = pca.components_
    loading_matrix = pd.DataFrame(loadings.T, columns=[f'PC{i+1}' for i in range(len(loadings))], index=data.columns)
    print("PCA Loadings:\n", loading_matrix)
    pc1_loadings = loading_matrix['PC1'].sort_values(ascending=False)
    loading_df = pd.DataFrame({'Feature': data_copy.columns, 'PC1 Loading': pc1_loadings.values})
    print("Features sorted by PC1 loadings:\n", loading_df)

  <img width="1280" height="612" alt="Feature NAMES loading in PC1" src="https://github.com/user-attachments/assets/9d9de9ba-1fd5-40aa-80c2-448a5598d1b3" />


  K-Means Algorithm as follows:
  
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

  Finding the best K-value using Elbow method:
  
 <img width="640" height="480" alt="Finalised Elbow Method" src="https://github.com/user-attachments/assets/c98db359-f90d-4e44-b39d-023f2bde4172" />

  In this project,best K value is 3 as per the image shown above,but practically ony 3 clusters group for music data is not acceptable so I have taken k-value as 7.

--->Evaluating the clusters

    score=silhouette_score(scaled_data,clusters)
    print(f'Silhouette Score: {score:.2f}')
    index=davies_bouldin_score(scaled_data,clusters)
    print(f'Davies-Bouldin Index: {index:.2f}')
      
The result is:
Silhouette Score: 0.39
Davies-Bouldin Index: 0.85

--->visualizing the clusters

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

<img width="640" height="480" alt="Kmeans clustering" src="https://github.com/user-attachments/assets/f3e30f0d-ab29-41db-bd0c-4024799a61e7" />


---> Next comes grouping the clusters

    # # Group by cluster to get mean values per feature
    cluster_mean = data_copy_1.groupby('clusters').mean()
    cluster_mean = cluster_mean.reset_index('clusters')
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

The result is

<img width="576" height="324" alt="cluster mean dataframe" src="https://github.com/user-attachments/assets/342c1b51-37f5-45b5-92de-8a76dd775c3b" />


--->Mappiing the clusters labels

    # Map the labels
    # cluster_mean['Cluster_Label'] = cluster_mean['clusters'].map(cluster_labels)
    
    # # --- Melt the data for visualization ---
    # cluster_melt = cluster_mean.melt(
    #     id_vars=['clusters', 'Cluster_Label'],
    #     value_vars=data_copy_1.columns.difference(['clusters']),
    #     var_name='Feature',
    #     value_name='Value')
    # # --- 4️⃣ Draw the line plot ---
    # plt.figure(figsize=(12, 6))
    # sns.lineplot(x='Feature', y='Value', hue='Cluster_Label',
    #              data=cluster_melt, marker='o')
    # plt.title("🎧 Cluster Feature Comparison (with Labels)", fontsize=14)
    # plt.xlabel("Audio Features")
    # plt.ylabel("Average Value")
    # plt.xticks(rotation=30)
    # plt.legend(title="Cluster Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    # plt.show()

<img width="1280" height="612" alt="cluste feature comparison final" src="https://github.com/user-attachments/assets/8af562b0-0be3-4a95-9ac3-81483c8a35f9" />

Then comes the heatmap visualization

      # # Heatmap visualization
      # plt.figure(figsize=(10,6))
      # sns.heatmap(
      #     cluster_mean.set_index('Cluster_Label')[data_copy_1.columns.difference(['clusters'])],
      #     cmap="YlGnBu", annot=True, fmt=".2f"
      # )
      # plt.title("🎨 Cluster Feature Heatmap", fontsize=14)
      # plt.xlabel("Features")
      # plt.ylabel("Cluster Type")
      # plt.show()
      
  <img width="1280" height="612" alt="cluster feature heatmap final" src="https://github.com/user-attachments/assets/cec83514-bc48-4759-a92e-18d98fb4c1a0" />

--->Tracking the popular songs in each clusters

    # top_by_popularity = data_copy_1.loc[
    #     data_copy_1.groupby('clusters')['popularity_songs'].idxmax()
    # ]
    # print("\nTop tracks by popularity:")
    # print(top_by_popularity)
   

<img width="576" height="324" alt="Top tracks" src="https://github.com/user-attachments/assets/20d330de-0d26-4c7a-badf-fbd07b92deb6" />




          
        
                




    

  




