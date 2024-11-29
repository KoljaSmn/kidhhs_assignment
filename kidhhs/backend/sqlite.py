import sqlite3 as db
import pandas as pd
import datetime
import os

from kidhhs.config.config import DATABASE_PATH

tweet_data_table_name = 'tweet'
sentiment_table = 'sentiment'


def setup_database():
    """
    Defines and creates the database scheme.
    :return:
    """
    if os.path.isfile(DATABASE_PATH):
        os.remove(DATABASE_PATH)
    con = db.connect(DATABASE_PATH)
    cur = con.cursor()

    # we use 2 tables: tweet and sentiment where sentiment only contains the sentiment score for each tweet and its id

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
                id INTEGER PRIMARY KEY, 
                sentiment REAL
                );""")


def write_covid_twitter_dataframe(covid_twitter_dataframe: pd.DataFrame, sentiment_dataframe):
    """
    writes tweet dataframe and corresponding sentiments to the sqlite database. Appends to the existing table.
    :param covid_twitter_dataframe:
    :param sentiment_dataframe:
    :return:
    """
    con = db.connect(DATABASE_PATH)
    covid_twitter_dataframe.to_sql(name=tweet_data_table_name, con=con, if_exists='append')
    sentiment_dataframe.to_sql(name=sentiment_table, con=con, if_exists='append')


def get_tweets_from_to(date_from: datetime.datetime, date_to: datetime.datetime):
    """
    Returns all tweets between the date_from and date_to.
    :param date_from:
    :param date_to:
    :return:
    """
    con = db.connect(DATABASE_PATH)
    df = pd.read_sql(f'select * from {tweet_data_table_name} where created_at BETWEEN ? and ?;', con=con,
                     params=(date_from, date_to))
    return df[['id', 'created_at', 'original_text']]


def get_tweets_and_sentiment_from_to(date_from: datetime.datetime, date_to: datetime.datetime):
    """
    Returns joined dataframe of tweet data and sentiments for all tweets between the given dates.
    :param date_from:
    :param date_to:
    :return:
    """
    con = db.connect(DATABASE_PATH)

    df = pd.read_sql(f"""SELECT * 
                         FROM {sentiment_table}
                         INNER JOIN {tweet_data_table_name}
                         ON {sentiment_table}.id={tweet_data_table_name}.id
                         WHERE {tweet_data_table_name}.created_at BETWEEN ? and ?;
                        """, con=con, params=(date_from, date_to))
    return df

def get_sentiment_from_to(date_from: datetime.datetime, date_to: datetime.datetime):
    """
    Gets all sentiments (and date) between the given dates.
    :param date_from:
    :param date_to:
    :return:
    """
    df = get_tweets_and_sentiment_from_to(date_from, date_to)
    return df[['created_at', 'sentiment']]
