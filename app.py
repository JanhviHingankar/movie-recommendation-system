import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def fetch_movie_details(movie_id):

    response = requests.get(
        f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=b8b022efaf5100c77357cd6a6d3dac5e&language=en-US'
    )

    data = response.json()

    if data.get('poster_path'):
        poster = "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    else:
        poster = "https://via.placeholder.com/500x750?text=No+Image"

    overview = data.get(
        'overview',
        'No description available.'
    )

    rating = data.get(
        'vote_average',
        'N/A'
    )

    return poster, overview, rating


movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))

movies = pd.DataFrame(movies_dict)

cv = CountVectorizer(
    max_features=5000,
    stop_words='english'
)

vectors = cv.fit_transform(
    movies['tags']
).toarray()

similarity = cosine_similarity(vectors)


def recommend(movie):

    movie_index = movies[
        movies['title'] == movie
    ].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []
    recommended_overviews = []
    recommended_ratings = []

    for i in movies_list:

        movie_id = movies.iloc[i[0]].movie_id

        recommended_movies.append(
            movies.iloc[i[0]].title
        )

        poster, overview, rating = fetch_movie_details(movie_id)

        recommended_posters.append(poster)

        recommended_overviews.append(overview)

        recommended_ratings.append(rating)

    return (
        recommended_movies,
        recommended_posters,
        recommended_overviews,
        recommended_ratings
    )


st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align:center; color:#FF4B4B;'>
        🎬 Movie Recommender System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h4 style='text-align:center; color:gray;'>
        Get personalized movie recommendations instantly
    </h4>
    """,
    unsafe_allow_html=True
)

st.write("")

selected_movie_name = st.selectbox(
    'Select a movie:',
    movies['title'].values
)

st.write("")

if st.button('Recommend'):

    names, posters, overviews, ratings = recommend(
        selected_movie_name
    )

    cols = st.columns(5)

    for idx, col in enumerate(cols):

        with col:

            st.image(
                posters[idx],
                width=180
            )

            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    font-size:16px;
                    font-weight:bold;
                    padding-top:10px;
                    min-height:60px;
                ">
                    {names[idx]}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div style='text-align:center; color:gold; font-weight:bold;'>
                     Rating: {ratings[idx]}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption(overviews[idx])