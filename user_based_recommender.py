####################################
# User-Based Collaborative Filtering
####################################
from numpy.lib import user_array


import pandas as pd

from item_based_recommender import user_movie_df

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)

#region Data Preparation
def create_user_movie_df():

    import pandas as pd
    movie = pd.read_csv('datasets/movie.csv')
    rating = pd.read_csv('datasets/rating.csv')
    df = movie.merge(rating, how='left', on='movieId')
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df



user_movie_df = create_user_movie_df()
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)
#endregion

#region
random_user_df = user_movie_df[user_movie_df.index == random_user]

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

user_movie_df.loc[user_movie_df.index == random_user,
                            user_movie_df.columns == "Schindler's List (1993)"]

#endregion

#region Other Users Watching The Same Movies

movies_watched_df = user_movie_df[movies_watched]
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)
user_movie_count[user_movie_count["movie_count"] == 33].count()
users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

# perc = len(movies_watched) * 60 / 100


#endregion

#region Determination of Similarity

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)], random_user_df[movies_watched]])

corr_df = pd.DataFrame(corr_df,columns=["corr"])

corr_df.index.names =["user_id_1", "user_id_2"]

corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values("corr", ascending=False)

top_users.rename(columns={"user_id": "userId"}, inplace=True)


rating = pd.read_csv('datasets/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId","rating"]], how="inner")

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
#endregion

#region Score Calculation

top_users_ratings["weighted_rating"] = top_users_ratings["corr"] * top_users_ratings["rating"]

recommendation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating": "mean"})

recommendation_df = recommendation_df.reset_index()

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

movie = pd.read_csv('datasets/movie.csv')
movies_to_be_recommend.merge(movie[["movieId", "title"]])

#endregion

#region Functionalization

def create_user_movie_df():

    import pandas as pd
    movie = pd.read_csv('datasets/movie.csv')
    rating = pd.read_csv('datasets/rating.csv')
    df = movie.merge(rating, how='left', on='movieId')
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

def user_based_recommender(random_user, user_movie_df, ratio=60, cor_th = 0.65, score=3.5):
    import pandas as pad
    random_user_df =  user_movie_df[user_movie_df.index == random_user]
    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    movies_watched_df = user_movie_df[movies_watched]
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)
    user_movie_count[user_movie_count["movie_count"] == 33].count()
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]
    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)], random_user_df[movies_watched]])
    corr_df = final_df.T.corr()
    corr_df = corr_df[~corr_df.index.duplicated()]
    corr_df = corr_df.unstack().sort_values()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df = pd.DataFrame(final_df.corr(), columns=["corr"])
    corr_df.index.names =["user_id_1", "user_id_2"]
    corr_df = corr_df.reset_index()
    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= cor_th)][["user_id_2", "corr"]].reset_index(drop=True)
    top_users = top_users.sort_values("corr", ascending=False)
    top_users.rename(columns={"user_id": "userId"}, inplace=True)
    rating = pd.read_csv('datasets/rating.csv')
    top_users_ratings = top_users.merge(rating[["userId", "movieId","rating"]], how="inner")
    top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
    top_users_ratings["weighted_rating"] = top_users_ratings["corr"] * top_users_ratings["rating"]
    recommendation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating": "mean"})
    recommendation_df = recommendation_df.reset_index()
    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score].sort_values("weighted_rating", ascending=False)
    movie = pd.read_csv('datasets/movie.csv')
    movies_to_be_recommend.merge(movie[["movieId", "title"]])
    return movies_to_be_recommend

random_user = int(pd.Series(user_movie_df.index).sample(1).values)
user_movie_df = create_user_movie_df()
user_based_recommender(random_user, user_movie_df)

#endregion
