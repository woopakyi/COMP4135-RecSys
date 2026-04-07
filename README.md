## Add the recommendation algorithm
The recommendation code is wired through `main.py`, but the actual algorithm work should be done in the template modules first.

Files to edit:
- `./flaskr/algo1.py` for FM
- `./flaskr/algo2.py` for SASRec
- `./flaskr/main.py` for wiring the final algorithm into the app

`algo1.py` and `algo2.py` are templates. When each implementation is finished, change the corresponding boolean in `main.py` from `False` to `True`:
- `algo1_is_done = False` -> `True` when FM is ready
- `algo2_is_done = False` -> `True` when SASRec is ready

`main.py` already routes requests through these modules, so you only need to turn the flags on after the templates are complete.

Important for algorithm developers:
- In `algo1.py`, only implement or modify `getRecommendationBy(...)` and `getLikedSimilarBy(...)`.
- In `algo2.py`, only implement or modify `getRecommendationBy(...)` and `getLikedSimilarBy(...)`.
- Other functions in those files are helper scaffolds for the default baseline and can be kept as-is unless your implementation needs different helpers.

## Create an environment

```
conda create -n lab3
conda activate lab3

```

## Install Python packages 

```
pip install --upgrade setuptools wheel pyquery
#conda install -c conda-forge scikit-surprise
pip install -r requirements.txt

```

## Run the project
```
flask --app flaskr run --debug
```

## About the Dataset
The dataset path is: ./flaskr/static/ml_data/

The ratings.csv file includes the following columns:
- userId: the IDs of users.  
- movieId: the IDs of movies.  
- rating: the rating given by the user to the movie, on a 5-star scale
- timestamp: the time when the user rated the movie, recorded in seconds since the epoch (as returned by time(2) function). A larger timestamp means the rating was made later.

You can use pandas to convert the timestamp to standard date and time. For example, 1717665888 corresponds to 2024-06-06 09:24:48.
```
import pandas as pd
timestamp = 1717665888
dt_str = pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')
print(dt_str)
```

## About Table IDs in Postgres
The `id` column is backed by a database sequence, not by the current number of rows.

That means:
- Deleting rows does not reset the next `id` value.
- If rows 1 to 9 are deleted manually, the next inserted row can still become `10`.
- This is normal in PostgreSQL and does not mean the table is broken.

If you need to reset IDs only in a local development database, use one of these approaches carefully:
- `TRUNCATE feedback RESTART IDENTITY;`
- `ALTER SEQUENCE feedback_id_seq RESTART WITH 1;`

Do not do this on shared or production data unless you are sure it is safe.

## Railway Cost Reduction Guide
To keep Railway usage low:
- Use the smallest practical service size.
- Remove unused services, workers, and background jobs.
- Turn off debug mode outside development.
- Avoid adding extra scheduled tasks unless they are necessary.
- Keep static files cacheable so the app does less work per request.
- Avoid frequent redeploys while testing small UI changes.
- Watch logs and database size, since heavy logging and stored data can increase usage.