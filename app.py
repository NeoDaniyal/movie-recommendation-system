import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

rating_stats = ratings.groupby("movieId").agg(
    rating_count = ('rating', 'count'),
    average_rating = ('rating', 'mean')
)

popular_movies = movies.merge(rating_stats, on="movieId").set_index('title')

similarity = np.random.rand(len(movies), len(movies))

def recommend_movies(movie_name):
    if movie_name not in movies['title'].values:
        return ["movie not found"]
    movie_idx = movies[movies['title'] == movie_name].index[0]
    distances = similarity[movie_idx]

    movie_list = sorted(list(enumerate(distances)), reverse=True, key =lambda x: x[1])[1:20]

    recommendation = []

    for i in movie_list:
        title = movies.iloc[i[0]]['title']
        recommendation.append(title)
    return recommendation[:5]


st.sidebar.title("Movie Recommendation")
st.sidebar.markdown("Select Your favorite Movie Here!")
Selected_movie = st.sidebar.selectbox("Chosse a Movie", movies['title'].values)
st.title("🎬 Movie Recommendation & Analytics Dashboard")
st.markdown("Welcome! Explore dataset statistics or generate algorithmic movie recommendations.")


st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Movies Available", value=movies.shape[0])
with col2:
    st.metric(label="Total User Rating logged", value=ratings.shape[0])
with col3:
    st.metric(label="Unique Platform Users", value=ratings['userId'].nunique())
    
st.write("---")

tab1, tab2 = st.tabs(["📈 Data Analytics", "🎥 Get Recommendations"])

with tab1:
    st.subheader("Movie Priority VS Average Ratings")

    fig = px.scatter(
        popular_movies,
        x="rating_count",
        y="average_rating",
            hover_name=popular_movies.index,
            labels={"rating_count": "Number of Ratings", "average_rating": "Average Rating (0-5)"},
            color= "average_rating",
            color_continuous_scale=px.colors.sequential.Viridis
            )
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("⭐ Top Rated Movies Exploration Table")
    st.dataframe(popular_movies.sort_values(by="average_rating", ascending=False).head(10))

    with tab2:
        st.subheader("Personalized Recommendation Engine")
        st.info(f"Currently Selected Target Movies:  **{Selected_movie}**")

        if st.button("Generate Recommendation: ", type="primary"):
            st.write("### Top 5 Recommended Matches For You:")

            recommendations= recommend_movies(Selected_movie)

            for mv in recommendations:
                st.success(mv)