# Movie Recommendation System

## Project Overview

The Movie Recommendation System suggests movies to users based on their preferences. Recommendation systems are widely used in platforms like Netflix, Amazon Prime, and YouTube to help users discover content they may like.

This project uses **machine learning techniques and similarity measures** to recommend movies based on movie features such as genres, keywords, cast, and overview.

The system analyzes movie data and recommends movies that are similar to the movie selected by the user.

---

## Dataset Used

This project uses two datasets:

1. **tmdb_5000_movies.csv**
2. **tmdb_5000_credits.csv**

These datasets contain important information about movies such as:

* Movie Title
* Genres
* Overview
* Cast
* Crew
* Keywords
* Movie ID

The datasets are merged using the **movie ID** to create a single dataset for analysis.

---

## Technologies Used

The following technologies and libraries are used in this project:

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Streamlit
* Pickle

---

## Project Workflow

### 1. Data Collection

The movie datasets (**tmdb_5000_movies.csv** and **tmdb_5000_credits.csv**) are loaded into the project.

### 2. Data Preprocessing

Data preprocessing is performed to prepare the dataset for the recommendation system.

Steps include:

* Merging both datasets
* Selecting relevant columns
* Removing missing values
* Converting text data into a usable format

---

### 3. Feature Engineering

Important movie features are combined into a single column called **tags**.
These features include:

* Genres
* Keywords
* Cast
* Crew
* Overview

Text processing techniques such as **tokenization and stemming** are applied to clean the data.

---

### 4. Text Vectorization

The textual data is converted into numerical format using **CountVectorizer** from Scikit-learn.

This creates a **vector representation** of each movie based on its tags.

---

### 5. Similarity Calculation

The similarity between movies is calculated using **Cosine Similarity**.

Cosine similarity measures how similar two movies are based on their feature vectors.

---

### 6. Movie Recommendation

When a user selects a movie, the system finds movies that have the **highest similarity score** and recommends the top similar movies.

---

### 7. Streamlit Web Application

A **Streamlit web application** is created to make the recommendation system interactive.

Users can:

* Select a movie
* Click the recommend button
* View recommended movies

---

## Installation and Setup

Follow the steps below to run the project locally.

### 1. Clone the repository

```id="gitclone"
git clone https://github.com/JanhviHingankar/movie-recommendation-system.git
```

### 2. Go to the project directory

```id="cdproject"
cd movie-recommendation-system
```

### 3. Install required libraries

```id="installreq"
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```id="runapp"
streamlit run app.py
```

The application will open in your browser.

---

## Project Structure

```id="structure"
movie-recommendation-system/

│
├── app.py
├── movies.pkl
├── similarity.pkl
├── requirements.txt
├── .env
├── .gitignore
├── README.md
└── datasets/
    ├── tmdb_5000_movies.csv
    └── tmdb_5000_credits.csv
```

---

## Conclusion

This project demonstrates how **machine learning and natural language processing techniques** can be used to build a movie recommendation system. By analyzing movie features and calculating similarity scores, the system can recommend movies that are similar to the user's selected movie.

Recommendation systems play a crucial role in improving user experience by helping users discover relevant content easily.

---

## Author

Janhvi Hingankar
