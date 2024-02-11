import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

movies = pickle.load(open("movies.pkl", "rb"))
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
movies['text_data'] = movies['key1'].fillna('')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['text_data'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
def get_recommendations(query, cosine_sim=cosine_sim):
    query_vec = tfidf_vectorizer.transform([query])
    cosine_similarities = linear_kernel(query_vec, tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_similarities))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 similar movies (excluding itself)
    movie_indices = [i[0] for i in sim_scores]
    return movies['original_title'].iloc[movie_indices]

st.title("Movie Recommender System")
query = st.text_input("Enter your query here",placeholder="Enter Actor, Movie or Director Name")

if st.button('Get Recommendations'):
    recommendations = get_recommendations(query)
    for movie_title in recommendations:
        st.write(movie_title)
        col1,col2,col3 = st.columns(3)