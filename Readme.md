# KIDHHS Assignment - WebApp for Sentiment Analysis Over Time

## Functionality
The web app allows the user to:
- Specify a date range to receive tweets from.
- Shows the tweets as a dataframe (id, date, text, sentiment)
- Shows plots for the sentiment (normalized/unnormalized) over time for different selectable time steps (day, week, month, year)
- Shows plots for the number of tweets
- Allows the user to enter a text and returns the corresponding sentiment score 



## Tools
This repository implements a web application to analyze the sentiment of texts over time.
Used tools: 
- Backend:
  - Sentiment Analysis: Huggingface (cardiffnlp/twitter-roberta-base-sentiment-latest) to analyze sentiment from
  - Dataset: https://www.kaggle.com/datasets/arunavakrchakraborty/covid19-twitter-dataset. 
  The csv-files are included in the project.
  - Database: SQLite with two tables
    - tweet (ID, original_text, lang, favorite_count, retweet_count, original_author, place, clean_tweet).
    - sentiment (ID, sentiment (float))
  - Flask: Offers API for get/post requests to (mainly) return data from the dataset.  
  The Port is set in kidhhs/config/config.py, 8080 by default.
- Frontend:
  - Streamlit: Runs on port 8081 by default (set in docker/compose.yaml).


## Project structure
The project is structured as follows:
- data
  - covid_twitter_dataset
    - ...csv
  - sqlite.db 
  - config.json: Saves settings (e.g., whether database has been initialized already)
- docker 
  - backend
    - Dockerfile for the backend app
    - requirements.txt for the backend app
  - frontend
    - Dockerfile for the frontend app
    - requirements.txt for the frontend app
- kidhhs: code
  - backend
    - api.py: Defines the flask API. The API offers four endpoints.
      - '/tweets' (POST) accepts two dates ('date_from', 'date_to') in json format and returns all tweets between the dates joined with the sentiment data as json.
      - '/sentiment' (POST) accepts two dates ('date_from', 'date_to') in json format and returns dataframes containing the date of the tweet and the sentiment as json (not used by the frontend currently).
      - '/textsentiment' (POST) accepts one text ({'text': 'some text'} in json format and returns {'sentiment': score} as json.
      - '/dpupdate' (get) returns the last update time of the database as json ({'last_update': timestamp}). Used by the frontend to determine whether the data cache is valid or whether it should be reloaded from backend.
      - ('/'  (get) returns 'Hello, World') to test whether the server is runnning. 
    - data_initialization.py: Called when starting the server, initializes the database. Iterates through data, computes sentiment.
    - dataset_loader.py: Called by data_initialization.py: Loads csv and returns pandas.DataFrame
    - main.py: starts the backend server. Starts the Flask API and starts process to initialize database.
    - sentiment_analysis.py: Runs sentiment analysis for batches of text or a single text.
    - sqlite.py: Defines database. Defines functions to read from the database.
  - config:
    - config.py: Allows configuration, ports paths etc.
  - frontend: 
    - app.py: Implements the streamlit app.


## Installation and Run
We use Docker compose to build and run two services (backend and frontend).
Run the following command to run the services from this working directory.:

```
BUILDKIT_PROGRESS=plain docker compose  -f docker/compose.yaml up
```

Then, the web app (frontend server) runs on localhost port 8081. 
The backend server runs on port 8080.

The repository already contains the created sqlite database. Remove it (or data/config.json) to recreate it.
This can take a while however.

When the backend is started for the first time, it will write the data from the csv-files to the database. The backend API is already working though and will return the batches of data that have already been written to the dataset.

