import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬"
)

@st.cache_data
def load_movies():
    movies = pd.read_csv("tmdb_5000_movies.csv")

    def extract_names(value):
        try:
            data = ast.literal_eval(value)
            return " ".join(item["name"] for item in data)
        except:
            return ""

    movies["genres_text"] = movies["genres"].apply(extract_names)
    movies["keywords_text"] = movies["keywords"].apply(extract_names)

    movies["overview"] = movies["overview"].fillna("")

    movies["combined_features"] = (
        movies["overview"] + " " +
        movies["genres_text"] + " " +
        movies["keywords_text"]
    )

    movies = movies.dropna(subset=["title"])
    movies = movies.reset_index(drop=True)

    return movies


@st.cache_resource
def create_model(features):
    vectorizer = TfidfVectorizer(stop_words="english")

    feature_matrix = vectorizer.fit_transform(features)

    return vectorizer, feature_matrix


movies = load_movies()

vectorizer, feature_matrix = create_model(
    movies["combined_features"].tolist()
)


def recommend(movie_title):
    movie_indices = movies.index[
        movies["title"] == movie_title
    ].tolist()

    if not movie_indices:
        return []

    idx = movie_indices[0]

    similarity_scores = cosine_similarity(
        feature_matrix[idx],
        feature_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[::-1][1:6]

    return movies.iloc[similar_indices]["title"].tolist()


# Streamlit UI

st.title("🎬 Movie Recommendation System")

st.write(
    "Discover movies similar to your favourite movie!"
)

movie_list = movies["title"].tolist()

selected_movie = st.selectbox(
    "Select a movie:",
    movie_list
)

if st.button("Recommend Movies"):
    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies")

    if recommendations:
        for movie in recommendations:
            st.write("🎥", movie)
    else:
        st.warning("No recommendations found.")