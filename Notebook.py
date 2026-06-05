#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


movies = pd.read_csv("movies.csv")
rating = pd.read_csv("ratings.csv")


# In[3]:


movies.head()


# In[4]:


rating.head()


# In[5]:


movies.shape


# In[6]:


rating.shape


# In[7]:


movies.info()


# In[8]:


rating.info()


# In[9]:


rating.describe()


# In[10]:


movie_rating = movies.merge(
    rating,
    on="movieId"
)


# In[11]:


movie_rating.head()


# In[12]:


avg_rating = movie_rating.groupby('title')['rating'].mean()
rating_count = movie_rating.groupby('title')['rating'].count()


# In[13]:


popular_movies = pd.DataFrame({
    "average_rating": avg_rating,
    "rating_count": rating_count
})

popular_movies


# In[14]:


popular_movies = popular_movies.sort_values(
    by="average_rating",
    ascending=False
)
popular_movies


# In[15]:


popular_movies = popular_movies[popular_movies["rating_count"] > 100]
popular_movies


# In[16]:


popular_movies.sort_values(
    by='average_rating',
    ascending=False
).head(10)


# In[17]:


fig = px.scatter(
    popular_movies,
    x='rating_count',
    y='average_rating',
    hover_name= popular_movies.index,
    title="Movie Popularity VS Rating"
)

fig.show()


# In[18]:


movies.head()


# In[19]:


movies['genres'] = movies['genres'].fillna('')


# In[20]:


cv = CountVectorizer(
    tokenizer= lambda x: x.split('|')
)


# In[21]:


genre_matrix = cv.fit_transform(
    movies['genres']
)


# In[22]:


similarity = cosine_similarity(
    genre_matrix
)


# In[23]:


def recommendation(movies_name):

    idx = movies[
        movies['title'] == movies_name
    ].index[0]
    sim_scores = list(
        enumerate(similarity[idx])
    )
    sim_scores = sorted(
        sim_scores,
        key= lambda x: x[1],
        reverse=True
    )
    sim_scores = sim_scores[1:6]
    for movie in sim_scores:
        print(
            movies.iloc[
                movie[0]
            ]['title']
        )


# In[24]:


recommendation('Toy Story (1995)')


# In[25]:


sample = similarity[:20, :20]

fig = px.imshow(
    sample,
    title="Movies Similarity Matrix"
)
fig.show()


# In[26]:


movie_rating.head()


# In[27]:


user_movie_matrix = movie_rating.pivot_table(
    index= 'userId',
    columns= 'title',
    values= 'rating'
)
user_movie_matrix.shape


# In[28]:


user_movie_matrix.head()


# In[29]:


user_movie_filled = user_movie_matrix.fillna(0)


# In[30]:


user_movie_filled.head()


# In[31]:


user_similarity = cosine_similarity(user_movie_filled)


# In[32]:


user_similarity_df = pd.DataFrame(
    user_similarity,
    index= user_movie_filled.index,
    columns=user_movie_filled.index
)

user_similarity_df.head()


# In[33]:


user_similarity_df[1].sort_values(
    ascending=False
).head(10)


# In[34]:


movie_user_matrix = movie_rating.pivot_table(
    index="title",
    columns="userId",
    values='rating'
)
movie_user_filled = movie_user_matrix.fillna(0)
movies_similarity = cosine_similarity(movie_user_filled)


# In[35]:


movies_stats = movie_rating.groupby(
    'title'
).agg({
    'rating': ['mean', 'count']
})


# In[36]:


movies_stats.head()


# In[37]:


movies_stats.columns = [
    'avg_rating',
    'rating_count'
]
movies_stats.head()


# In[42]:


def recommend_movies(movie_name):

    if movie_name not in movies['title'].values:
        print("Movie not fount")
        return
    movie_idx = movies[
        movies['title'] == movie_name
    ].index[0]
    distances = similarity[movie_idx]

    movie_list = sorted(
        list(enumerate(distances)),

        reverse=True,

        key= lambda x: x[1]
    )[1:20]
    recommendation = []
    for i in movie_list:
        title = movies.iloc[
            i[0]
        ]['title']
        recommendation.append(
            title
        )
    return recommendation[:5]


# In[44]:


recs = recommend_movies(
    'Toy Story (1995)'
)

for movie in recs:
    print(movie)


# In[ ]:




