import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- Movie ratings (tiny sample dataset) ---
data = {
    'movie': [
        'Toy Story',
        'Jumanji',
        'Grumpier Old Men',
        'Waiting to Exhale',
        'Father of the Bride II'
    ],
    'user1': [5, 4, 1, 0, 0],
    'user2': [4, 5, 1, 0, 0],
    'user3': [5, 4, 0, 2, 1],
    'user4': [0, 0, 5, 5, 4],
}

df = pd.DataFrame(data)
df.set_index('movie', inplace=True)

# --- Compute similarity between movies ---
similarity_matrix = cosine_similarity(df)
similarity_df = pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)

# --- Function to recommend similar movies ---
def recommend(movie_name, top_n=3):
    if movie_name not in similarity_df.index:
        return "Movie not found."

    scores = similarity_df[movie_name].sort_values(ascending=False)
    scores = scores.drop(movie_name)  # remove same movie
    return scores.head(top_n)

# --- Demo Output ---
print("Movies in our system:")
for movie in similarity_df.index:
    print("-", movie)

print("\nRecommendations for 'Toy Story':")
print(recommend("Toy Story"))
