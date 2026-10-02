# Movie recommendation system

Content-based movie recommender. Type an actor, director or movie name and get nine similar movies, each with its poster and a link to TMDB.

## How it works

- **Data:** the TMDB 5000 movies and credits datasets from Kaggle (`tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`).
- **Features:** each movie's title, genres, keywords, overview, production companies, release date, tagline, cast and crew are combined into one string, then cleaned with NLTK (punctuation removed, tokens stemmed and lemmatized). `Movie_Recommendation.ipynb` builds this and saves it as `movies.pkl`.
- **Similarity:** TF-IDF vectors with English stop words removed, ranked by cosine similarity to the query.
- **App:** `app.py` is a Streamlit page that fetches posters from the TMDB API.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Poster lookups need a TMDB API key (free from themoviedb.org). Put it in `.streamlit/secrets.toml` as `TMDB_API_KEY = "your-key"`, or set a `TMDB_API_KEY` environment variable. Without a key the app still works and shows a placeholder image instead of posters.

## Limits

It matches words, not meaning, so a query only finds movies that share vocabulary with it. There are no ratings or user history involved.
