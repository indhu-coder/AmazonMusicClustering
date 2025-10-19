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

  K-Means Algorithm as follows:
      #clustering the data using Kmeans
  
    k_range= range(2,16) 
    sse=[]
    silhouette_scores=[]
    for k in k_range:
        kmeans=KMeans(n_clusters=k,random_state=42)
        clusters = kmeans.fit_predict(scaled_data_1)
        sse.append(kmeans.inertia_)
        score=silhouette_score(scaled_data_1,clusters)
        silhouette_scores.append(score)
        print(f'Silhouette Score for k={k}: {score}')

  Finding the best K-value using Elbow method:

  <img width="640" height="480" alt="Elbow Method" src="https://github.com/user-attachments/assets/f524e050-02e3-4e3d-87c5-403474974672" />

  

        




    

  




