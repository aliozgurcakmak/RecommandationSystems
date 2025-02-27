################################
# Content-Based Recommendation
################################


# Developing Recommendations Based on Movie Reviews

## Creating TF-IDF Matrix
## Creating Cosine Similiarity Matrix
## Making Recommendations According to Similarities
## Preparing Script


#region Creating TF-IDF Matrix

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("movies_metadata.csv", low_memory=False)
df.head()
df.shape

df["overview"].head()

tfidf = TfidfVectorizer(stop_words='english')

df[df["overview"].isnull()]
df["overview"] = df["overview"].fillna("")

tfidf_matrix = tfidf.fit_transform(df["overview"])
tfidf_matrix.shape

df["title"].shape

tfidf_matrix.toarray()
#endregion

#region Creating Cosine Similiarity Matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim.shape
cosine_sim[1]
#endregion

#region Making Recommendations According to Similarities

indices = pd.Series(df.index, index=df["title"])
indices.index.value_counts()

indices = indices[~indices.index.duplicated(keep="last")]
indices["Cinderella"]
indices["Sherlock Holmes"]

movie_index = indices["Sherlock Holmes"]

cosine_sim[movie_index]

similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["Score"])

movie_indices = similarity_scores.sort_values(by="Score", ascending=False)[1:11].index

df["title"].iloc[movie_indices]
#endregion

#region Preparing Script

def content_based_recommender(title, cosine_sim, dataframe):
    indices = pd.Series(dataframe.index, index=dataframe["title"])
    indices = indices[~indices.index.duplicated(keep="last")]
    movie_index = indices[title]
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    movie_indices = similarity_scores.sort_values(by="score", ascending=False)[1:11].index
    return dataframe["title"].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)
content_based_recommender("Toy Story", cosine_sim, df)
content_based_recommender("The Matrix", cosine_sim, df)
content_based_recommender("The Shawshank Redemption", cosine_sim, df)
content_based_recommender("V for Vendetta", cosine_sim, df)
content_based_recommender("Fight Club", cosine_sim, df)
content_based_recommender("Pixels", cosine_sim, df)
content_based_recommender("The Godfather", cosine_sim, df)

def calculate_cosine_sim(dataframe):
 tfidf = TfidfVectorizer(stop_words='english')
 dataframe["overview"] = dataframe["overview"].fillna("")
 tfidf_matrix = tfidf.fit_transform(dataframe["overview"])
 cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
 return cosine_sim
#endregion