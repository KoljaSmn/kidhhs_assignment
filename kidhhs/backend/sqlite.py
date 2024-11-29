import sqlite3 as db
import pandas as pd
import datetime
import os

from kidhhs.config.config import DATABASE_PATH

tweet_data_table_name = 'tweet'
sentiment_table = 'sentiment'


def setup_database():
    if os.path.isfile(DATABASE_PATH):
        os.remove(DATABASE_PATH)
    con = db.connect(DATABASE_PATH)
    cur = con.cursor()
    cur.execute(f"""CREATE TABLE {tweet_data_table_name} (
                id INTEGER PRIMARY KEY, 
                created_at TEXT, 
                original_text TEXT,
                lang TEXT,
                favorite_count INTEGER,
                retweet_count INTEGER,
                original_author TEXT,
                place TEXT,
                clean_tweet TEXT
                );""")
    cur.execute(f"""CREATE TABLE {sentiment_table} (
                id INTEGER NOT NULL UNIQUE, 
                sentiment REAL
                );""")


def write_covid_twitter_dataframe(covid_twitter_dataframe: pd.DataFrame, sentiment_dataframe):
    con = db.connect(DATABASE_PATH)
    covid_twitter_dataframe.to_sql(name=tweet_data_table_name, con=con, if_exists='append')
    sentiment_dataframe.to_sql(name=sentiment_table, con=con, if_exists='append')


def get_tweets_from_to(date_from: datetime.datetime, date_to: datetime.datetime):
    con = db.connect(DATABASE_PATH)
    df = pd.read_sql(f'select * from {tweet_data_table_name} where created_at BETWEEN ? and ?;', con=con,
                     params=(date_from, date_to))
    return df[['id', 'created_at', 'original_text']]


def get_tweets_and_sentiment_from_to(date_from: datetime.datetime, date_to: datetime.datetime):
    con = db.connect(DATABASE_PATH)

    df = pd.read_sql(f"""SELECT * 
                         FROM {sentiment_table}
                         INNER JOIN {tweet_data_table_name}
                         ON {sentiment_table}.id={tweet_data_table_name}.id
                         WHERE {tweet_data_table_name}.created_at BETWEEN ? and ?;
                        """, con=con, params=(date_from, date_to))
    return df

def get_sentiment_from_to(date_from: datetime.datetime, date_to: datetime.datetime):
    df = get_tweets_and_sentiment_from_to(date_from, date_to)
    return df[['created_at', 'sentiment']]


date_from = datetime.datetime(2020, 3, 1)
date_to = datetime.datetime(2020, 5, 1)
