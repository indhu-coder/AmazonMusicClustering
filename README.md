Amazon strives to be Earth’s most customer-centric company, Earth’s best employer, and Earth’s safest place to work. 
Customer reviews, 1-Click shopping, personalized recommendations, Prime, Music, Fulfillment by Amazon, AWS, Kindle Direct Publishing, Kindle, Career Choice, Fire tablets, Fire TV, Amazon Echo, Alexa, Just Walk Out technology, Amazon Studios, and The Climate Pledge are some of the things pioneered by Amazon.

Problem Statement:

  With millions of songs available on platforms like Amazon, manually categorizing tracks into genres is impractical. 
  The goal of this project is to automatically group similar songs based on their audio characteristics using clustering techniques. 
  By analyzing patterns in features such as tempo, energy, danceability, and more, learners will develop a model that organizes songs into meaningful clusters, potentially representing different musical genres or moods—without any prior labels.

Dataset features expalanation:
 
| **Feature Name**     | **Description**                                                                |
| -------------------- | ------------------------------------------------------------------------------ |
| **duration_ms**      | Average song length in milliseconds (e.g., 12.2 ≈ 2–3 minutes).                |
| **danceability**     | Measures how suitable a track is for dancing (scale: 0–1).                     |
| **energy**           | Describes the intensity or activity level of a song (higher = more energetic). |
| **loudness**         | Overall average loudness in decibels (lower values = quieter).                 |
| **speechiness**      | Detects the presence of spoken words (higher = more like rap or speech).       |
| **acousticness**     | Likelihood the track is acoustic (1.0 = fully acoustic, 0.0 = electronic).     |
| **instrumentalness** | Probability that a track contains no vocals (1.0 = instrumental only).         |
| **valence**          | Describes the musical positivity or mood (1.0 = happy, 0.0 = sad).             |
| **tempo**            | The speed of the track in beats per minute (BPM).                              |
| **popularity_songs** | The average popularity score of the song (based on user/stream data).          |
| **followers**        | Average number of followers of the artist(s) associated with the track.        |
| **year**             | Average release year of the track.                                             |
| **explicit**         | Indicates explicit content (1 = explicit, 0 = non-explicit).                   |
| **mode**             | Musical mode: 1 = major (happy tone), 0 = minor (sad tone).                    |
| **genres**           | Encoded numerical label for the genre (e.g., Pop=3, Rock=4, etc.).             |



