# 🎬 Movie Recommendation System (Streamlit Web App)

A simple and interactive **Movie Recommender Web App** built using **Python, Streamlit, Pandas, and Scikit-learn**.  
This project uses **Item-Based Collaborative Filtering** and calculates **cosine similarity** between movies based on user ratings from the **MovieLens 100k dataset**.

The web app allows users to:
- Select a movie from a dropdown list
- Choose how many recommendations they want
- View top similar movies instantly

Deployed using **Streamlit Cloud**.

---

## 🚀 Features
- User-friendly web interface built with **Streamlit**
- Movie dropdown selector
- Adjustable number of recommendations (1–20)
- Automatic dataset download (MovieLens 100k)
- Cosine similarity-based recommendations
- Clean and minimal UI
- Deployable online via GitHub + Streamlit Cloud

---

## 🧠 How It Works

### 1. **Dataset**
The app downloads the **MovieLens 100k** dataset:
- `u.data` → user ratings  
- `u.item` → movie titles  

### 2. **Preprocessing**
- Creates a **movie-user rating matrix**
- Fills missing ratings with 0
- Converts movie IDs to movie titles

### 3. **Cosine Similarity**
We compute **item–item similarity**:
```python
similarity = cosine_similarity(movie_user_matrix)
