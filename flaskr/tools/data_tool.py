import os
import pandas as pd


def loadData():
    return getMovies(), getGenre(), getRates()


# movieId,title,year,overview,cover_url,genres
def getMovies():
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/movie_info.csv"
    df = pd.read_csv(path)
    df['genres'] = df.genres.str.split('|')
    # Sanitize NaN values so jsonify never outputs bare `NaN` (invalid JSON).
    # The new getMoviesByGenres ranks by cosine similarity, which can surface niche
    # movies (e.g. Cheburashka) that have missing overview/year in the CSV.
    # Python's json.dumps serialises float nan as the literal NaN, which browsers
    # reject in response.json(), silently breaking all dynamic section updates.
    df['year'] = pd.array([None if pd.isna(x) else int(x) for x in df['year']], dtype=object)
    df['overview'] = df['overview'].fillna('')
    df['cover_url'] = df['cover_url'].fillna('')

    return df


# A list of the genres.
def getGenre():
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/genre.csv"
    df = pd.read_csv(path, delimiter="|", names=["name", "id"])
    df.set_index('id')
    return df


# user id, item id, rating, timestamp
def getRates():
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/ml_data/ratings.csv"
    df = pd.read_csv(path, delimiter=",", header=0, names=["userId", "movieId", "rating", "timestamp"])
    df = df.drop(columns='timestamp')
    df = df[['userId', 'movieId', 'rating']]

    return df


# itemID | userID | rating
def ratesFromUser(rates):
    itemID = []
    userID = []
    rating = []

    for rate in rates:
        items = rate.split("|")
        userID.append(int(items[0]))
        itemID.append(int(items[1]))
        rating.append(int(items[2]))

    ratings_dict = {
        "userId": userID,
        "movieId": itemID,
        "rating": rating,
    }

    return pd.DataFrame(ratings_dict)