import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
#Algorithm Used : Item-Based Collaborative Filtering using cosine Similarity
# Page config
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# 🎨 Custom CSS (same but slightly improved spacing)
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #00BFFF;
}
.subtitle {
    text-align: center;
    color: #CCCCCC;
    margin-bottom: 25px;
}
.card {
    padding: 15px;
    border-radius: 15px;
    background: linear-gradient(135deg, #1f4037, #99f2c8);
    color: black;
    text-align: center;
    margin: 10px;
    font-weight: bold;
}
.selected {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg, #ff512f, #dd2476);
    color: white;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
.topcard {
    padding: 12px;
    border-radius: 12px;
    background: linear-gradient(135deg, #f7971e, #ffd200);
    color: black;
    text-align: center;
    margin: 5px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Discover movies similar to your favorite one</div>', unsafe_allow_html=True)

# Load data
ratings = pd.read_csv('u.data', sep='\t', names=['user_id','movie_id','rating','timestamp'])
movies = pd.read_csv('u.item', sep='|', encoding='latin-1', header=None)
movies = movies[[0,1]]
movies.columns = ['movie_id','title']

data = pd.merge(ratings, movies, on='movie_id')

# Sidebar
st.sidebar.header("📊 Dataset Info")
st.sidebar.write("👤 Users:", data['user_id'].nunique())
st.sidebar.write("🎬 Movies:", data['movie_id'].nunique())
st.sidebar.write("⭐ Ratings:", len(data))

# Pivot + similarity
movie_matrix = data.pivot_table(index='title', columns='user_id', values='rating').fillna(0)
similarity = cosine_similarity(movie_matrix)

# Recommendation function
def recommend(movie_name, n=6):
    index = movie_matrix.index.get_loc(movie_name)
    distances = similarity[index]
    sorted_movies = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:]
    return [(movie_matrix.index[i[0]], i[1]) for i in sorted_movies[:n]]

# 🔝 Top Rated Movies
top_movies = data.groupby('title')['rating'].mean().sort_values(ascending=False).head(10)

# 🔥 Popular Movies (NEW small feature)
popular_movies = data.groupby('title').size().sort_values(ascending=False).head(10)

# Search
search = st.text_input("🔍 Search Movie")

movie_list = movie_matrix.index.tolist()
if search:
    movie_list = [m for m in movie_list if search.lower() in m.lower()]

# Select movie
selected_movie = st.selectbox("🎯 Select Movie", movie_list)

# Slider
num = st.slider("📌 Number of Recommendations", 3, 12, 6)

st.markdown("<br>", unsafe_allow_html=True)

# Buttons (clean alignment)
col1, col2, col3 = st.columns([1,1,1])

with col1:
    recommend_btn = st.button("🚀 Recommend")

with col2:
    show_top = st.button("⭐ Top Rated")

with col3:
    show_popular = st.button("🔥 Popular")

# 👉 Top Rated
if show_top:
    st.subheader("⭐ Top Rated Movies")
    cols = st.columns(5)
    for i, movie in enumerate(top_movies.index):
        cols[i % 5].markdown(
            f'<div class="topcard">⭐ {movie}</div>',
            unsafe_allow_html=True
        )

# 👉 Popular Movies
if show_popular:
    st.subheader("🔥 Popular Movies")
    cols = st.columns(5)
    for i, movie in enumerate(popular_movies.index):
        cols[i % 5].markdown(
            f'<div class="topcard">🔥 {movie}</div>',
            unsafe_allow_html=True
        )

# 👉 Recommendations
if recommend_btn:
    recs = recommend(selected_movie, num)

    st.markdown("<br>", unsafe_allow_html=True)

    # Selected movie
    st.markdown(
        f'<div class="selected">🎯 {selected_movie}</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Explanation (VERY IMPORTANT 🔥)
    st.info(f"💡 Recommendations are based on users who liked '{selected_movie}' also liked similar movies.")

    # Recommended movies
    st.subheader("🎬 Recommended Movies")

    cols = st.columns(3)

    for i, (movie, score) in enumerate(recs):
        cols[i % 3].markdown(
            f'<div class="card">🎬 {movie}<br>⭐ {round(score,2)}</div>',
            unsafe_allow_html=True
        )