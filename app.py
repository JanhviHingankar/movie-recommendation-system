import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- FETCH MOVIE POSTER ---------------- #

def fetch_poster(movie_id):

    response = requests.get(
        f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=b8b022efaf5100c77357cd6a6d3dac5e&language=en-US'
    )

    data = response.json()

    if 'poster_path' in data and data['poster_path'] is not None:
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

    return "https://via.placeholder.com/500x750?text=No+Image"


# ---------------- LOAD MOVIE DATA ---------------- #

movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))

movies = pd.DataFrame(movies_dict)


# ---------------- GENERATE SIMILARITY ---------------- #

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)


# ---------------- RECOMMEND FUNCTION ---------------- #

def recommend(movie):

    movie_index = movies[movies['title'] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    recommended_movies_posters = []

    for i in movies_list:

        movie_id = movies.iloc[i[0]].movie_id

        recommended_movies.append(
            movies.iloc[i[0]].title
        )

        recommended_movies_posters.append(
            fetch_poster(movie_id)
        )

    return recommended_movies, recommended_movies_posters


# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align: center; color: #FF4B4B;'>
        🎬 Movie Recommender System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h4 style='text-align: center; color: gray;'>
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


# ---------------- BUTTON ---------------- #

if st.button('Recommend'):

    names, posters = recommend(selected_movie_name)

    cols = st.columns(5)

    for idx, col in enumerate(cols):

        with col:

            st.image(
                posters[idx],
                use_container_width=True
            )

            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    font-size:16px;
                    font-weight:bold;
                    padding-top:10px;
                    min-height:70px;
                ">
                    {names[idx]}
                </div>
                """,
                unsafe_allow_html=True
            )