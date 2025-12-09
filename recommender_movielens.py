# recommender_movielens.py
import os
import urllib.request
import zipfile
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_ZIP = "ml-100k.zip"
DATA_DIR = "ml-100k"

# 1) Download MovieLens 100k if not present
if not os.path.exists(DATA_DIR):
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    print("Downloading MovieLens 100k. This may take a minute...")
    urllib.request.urlretrieve(url, DATA_ZIP)
    print("Unzipping...")
    with zipfile.ZipFile(DATA_ZIP, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Done.")

# 2) Load ratings and movies
# ratings file format: user id | item id | rating | timestamp
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv(os.path.join(DATA_DIR, 'u.data'), sep='\t', names=r_cols, encoding='latin-1')

# movies file format in u.item, pipe separated — movie id | movie title | ...
m_cols = ['movie_id', 'title'] + [f'f{i}' for i in range(22)]
movies = pd.read_csv(os.path.join(DATA_DIR, 'u.item'), sep='|', names=m_cols, encoding='latin-1', usecols=[0,1])

# 3) Build movie-user matrix (movies as rows, users as columns)
# take a pivot table and fill NaN with 0
movie_user = ratings.pivot_table(index='movie_id', columns='user_id', values='rating', fill_value=0)

# map movie_id -> title
movie_user.index = movie_user.index.map(lambda mid: movies.loc[movies.movie_id==mid, 'title'].values[0])

# 4) compute item-item similarity (cosine)
print("Computing similarity matrix (may take ~10-30 seconds)...")
sim_matrix = cosine_similarity(movie_user)
sim_df = pd.DataFrame(sim_matrix, index=movie_user.index, columns=movie_user.index)
print("Done computing similarity.")

# 5) recommendation function
def recommend(movie_title, top_n=5):
    if movie_title not in sim_df.index:
        return f"Movie '{movie_title}' not found. Try one of: {list(sim_df.index)[:10]}"
    scores = sim_df[movie_title].sort_values(ascending=False).drop(movie_title)
    return list(zip(scores.index[:top_n], scores.values[:top_n].round(3)))

# Demo: recommend
if __name__ == "__main__":
    while True:
        q = input("\nType a movie name (or 'exit'): ").strip()
        if q.lower() in ('exit', 'quit'): break
        print(recommend(q, top_n=5))

