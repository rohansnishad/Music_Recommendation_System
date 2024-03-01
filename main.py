import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Step 1: Read the Excel file
def read_dataset(path):
    return pd.read_excel(path)


# Step 2: Process the Data
def process_data(data):
    # For simplicity, we're going to create a 'combined' column that includes all the information
    # You might have more complex feature engineering here
    data['combined'] = data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    return data


# Step 3: Recommendation Logic
def make_recommendations(data, song, num_recommendations=5):
    # Using TF-IDF Vectorizer to convert text data into a matrix of token counts
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['combined'])

    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get pairwise similarity scores for all songs with that song
    sim_scores = list(enumerate(cosine_sim[song]))

    # Sort the songs based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top-n most similar songs
    sim_scores = sim_scores[1:num_recommendations + 1]

    # Get the song indices
    song_indices = [i[0] for i in sim_scores]

    # Return the top-n most similar songs
    return data['title'].iloc[song_indices]


# Execute the recommendation script
if __name__ == "__main__":
    data_path = 'SpotifyAudioFeaturesNov2018.xlsx'
    data = read_dataset(data_path)
    processed_data = process_data(data)

    # Let's assume you want recommendations similar to the first song in your dataset
    recommendations = make_recommendations(processed_data, song=0, num_recommendations=5)
    print("Recommendations:")
    print(recommendations)
