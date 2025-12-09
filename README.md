# Movie Recommendation System

This is a simple Movie Recommendation System built using Python, Streamlit, Pandas, and Scikit-learn.  
The system recommends similar movies based on cosine similarity between movie rating patterns.  
The dataset used is MovieLens 100k, which is downloaded automatically by the application.

---

## Features
- Simple web interface using Streamlit  
- Select a movie from a dropdown  
- Choose how many recommendations you want  
- Shows top similar movies  
- Uses item-based collaborative filtering  
- Easy to deploy and run  

---

## Flowchart (How the system works)

                   +------------------------+
                   |      User selects      |
                   |       a movie          |
                   +-----------+------------+
                               |
                               v
                   +------------------------+
                   | Load MovieLens dataset |
                   |   (movie ratings)      |
                   +-----------+------------+
                               |
                               v
                   +------------------------+
                   | Create movie-user      |
                   | rating matrix          |
                   +-----------+------------+
                               |
                               v
                   +------------------------+
                   | Compute cosine         |
                   | similarity between     |
                   | all movies             |
                   +-----------+------------+
                               |
                               v
                   +------------------------+
                   | Sort similarity scores |
                   | and take top results   |
                   +-----------+------------+
                               |
                               v
                   +------------------------+
                   | Display recommended    |
                   | movies in Streamlit    |
                   +------------------------+

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/MovieRecommendationSystem.git
cd MovieRecommendationSystem
