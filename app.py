# app.py -- Streamlit Movie Recommender (item-item cosine)
import os
import urllib.request
import zipfile
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.set_page_config(page_title="Movie Recommender", layout="centered")

@st.cache_data(show_spinner=False)
def download_movielens():
    DATA_ZIP = "ml-100k.zip"
    DATA_DIR = "ml-100k"
    if not os.path.exists(DATA_DIR):
        url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
        urllib.request.urlretrieve(url, DATA_ZIP)
        with zipfile.ZipFile(DATA_ZIP, 'r') as z:
            z.extractall(".")
    return DATA_DIR

@st.cache_data(show_spinner=False)
def load_data():
    data_dir = download_movielens()
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(os.path.join(data_dir, 'u.data'), sep='\t', names=r_cols, encoding='latin-1')
    m_cols = ['movie_id', 'title'] + [f'f{i}' for i in range(22)]
    movies = pd.read_csv(os.path.join(data_dir, 'u.item'), sep='|', names=m_cols, encoding='latin-1', usecols=[0,1])
    # pivot: movies x users
    movie_user = ratings.pivot_table(index='movie_id', columns='user_id', values='rating', fill_value=0)
    # map ids to titles
    movie_user.index = movie_user.index.map(lambda mid: movies.loc[movies.movie_id==mid, 'title'].values[0])
    return movie_user

@st.cache_data(show_spinner=False)
def compute_similarity(movie_user):
    sim_matrix = cosine_similarity(movie_user)
    sim_df = pd.DataFrame(sim_matrix, index=movie_user.index, columns=movie_user.index)
    return sim_df

def recommend(sim_df, movie_title, top_n=5):
    if movie_title not in sim_df.index:
        return []
    scores = sim_df[movie_title].sort_values(ascending=False).drop(movie_title)
    top = list(zip(scores.index[:top_n], scores.values[:top_n].round(3)))
    return top

# ------------------ Streamlit UI ------------------
st.title("🎬 Movie Recommender")
st.write("Pick a movie and I will suggest movies that similar users liked.")

# Load data (cached)
with st.spinner("Loading data (download may happen once)..."):
    movie_user = load_data()
    sim_df = compute_similarity(movie_user)

# UI controls
col1, col2 = st.columns([3,1])
with col1:
    movie_choice = st.selectbox("Choose a movie", options=movie_user.index.sort_values(), index=0)
with col2:
    top_n = st.slider("How many?", min_value=1, max_value=20, value=7)

if st.button("Show recommendations"):
    with st.spinner("Computing recommendations..."):
        recs = recommend(sim_df, movie_choice, top_n=top_n)
    if not recs:
        st.warning("Movie not found. Try a different title.")
    else:
        df = pd.DataFrame(recs, columns=["Recommended Movie", "Score"])
        st.success(f"Top {len(df)} movies similar to **{movie_choice}**")
        st.dataframe(df.reset_index(drop=True))
        st.markdown("**Recommendations (top → best):**")
        for i, (m, s) in enumerate(recs, start=1):
            st.write(f"{i}. {m} — score: {s}")

st.markdown("---")
st.caption("Data: MovieLens 100k. Similarity: cosine on movie×user ratings.")
