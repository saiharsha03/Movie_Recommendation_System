import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import requests

movies = pickle.load(open("movies.pkl", "rb"))
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['KEY'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(query, cosine_sim=cosine_sim):
    query_vec = tfidf_vectorizer.transform([query])
    cosine_similarities = linear_kernel(query_vec, tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_similarities))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:10] 
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = [(movies.iloc[index]['movie_id'], movies.iloc[index]['title']) for index in movie_indices]
    return recommended_movies

def get_movie_tmdb_link(movie_id):
    return f"https://www.themoviedb.org/movie/{movie_id}"

def get_movie_poster(movie_id):
    api_key = 'd2a6d95263679c62dda7c1677df1b05d'
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        poster_path = data['poster_path']
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    return "https://upload.wikimedia.org/wikipedia/commons/a/ac/No_image_available.svg"

st.title("Movie Recommender System")
query = st.text_input("Enter your query here",placeholder="Enter Actor, Movie or Director Name")

if st.button('Get Recommendations'):
    recommendations = get_recommendations(query)
    num_columns = 3
    cols = st.columns(num_columns)
    with st.spinner('Loading...'):
        for i, (movie_id, movie_title) in enumerate(recommendations):
            poster_url = get_movie_poster(movie_id)
            if poster_url:
                col_index = i % num_columns
                container = cols[col_index].container()
                container.image(poster_url, caption=movie_title, use_column_width=True)
                container.markdown(f"[View on TMDB]({get_movie_tmdb_link(movie_id)})")
                