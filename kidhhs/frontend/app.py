import streamlit as st
from datetime import datetime
import altair as alt
import pandas as pd
import requests

import sys
import os

sys.path += ['.']

from kidhhs.config.config import BACKEND_PORT, SENTIMENT_POST_NAME, TEXT_SENTIMENT, TWEETS_AND_SENTIMENT_POST_NAME, \
    BACKEND_URL, DB_UPDATE


@st.cache_data
def _load_tweet_data(date_from, date_to, last_db_update):
    """
    Loads tweet data between the given dates from the backend.

    :param date_from:
    :param date_to:
    :return:
    """
    # specify url using information from kidhhs/config/config.py
    url = f"{BACKEND_URL}:{BACKEND_PORT}/{TWEETS_AND_SENTIMENT_POST_NAME}"
    # dates as json objects
    obj = {'date_from': date_from, 'date_to': date_to}
    # send post request to the backend
    request_answer = requests.post(url, json=obj).json()
    # convert json answer to pd.DataFrame
    tweets_df = pd.DataFrame.from_dict(request_answer)
    return tweets_df


def _get_last_db_update():
    """
    Returns the last update time of the database from backend.
    Used to determine whether the data cache is valid (or the database changed).
    :return:
    """
    # specify url using information from kidhhs/config/config.py
    url = f"{BACKEND_URL}:{BACKEND_PORT}/{DB_UPDATE}"
    # send get request to the backend
    request_answer = requests.get(url).json()
    return request_answer['last_update']


def init_frontend():
    """
    Initializes the streamlit page.
    :return:
    """

    # use wide page layout
    st.set_page_config(layout="wide")

    # page header
    st.header("Covid Twitter Sentiment Analysis Over Time")

    # subheader. Dates can be selected here.
    st.subheader('Date Selection')
    st.text(
        'Here you can choose the date range to consider tweets from. The selection will be used in all following subsections.')

    # generate columns for date selection (from, to)
    tweets_date_col1, tweets_date_col2 = st.columns(2)
    date_input_count = 0
    with tweets_date_col1:
        from_date_input_tweets = st.date_input('From date', value=datetime(2019, 12, 1), key=date_input_count)
        date_input_count += 1
    with tweets_date_col2:
        to_date_input_tweets = st.date_input('To date', value=datetime(2021, 12, 31), key=date_input_count)
        date_input_count += 1

    # load the tweet data, the _load_tweet_data function is cached
    # but will be reloaded when either the dates or the database changed
    tweets_df = _load_tweet_data(str(from_date_input_tweets), str(to_date_input_tweets), _get_last_db_update())

    # section to display the tweets in
    st.subheader("Tweets")

    # the tweets are displayed in a dataframe, the dataframe can be collapsed/expanded
    with st.expander("Tweet Sentiments"):
        st.dataframe(tweets_df, use_container_width=True)

    # analysis of sentiments over time
    st.subheader("Sentiment Data Over Time")

    def get_group_by_dataframe(dataframe, selectbox_key):
        """
        Function that groups tweet dataframe using the date and the selected time_step_option. E.g. day or Week.
        :param dataframe:
        :param selectbox_key:
        :return:
        """
        time_step_options = ['Day', 'Week', 'Month', 'Year']

        time_step_select_box = st.selectbox('Figure Group By', time_step_options,
                                            index=2,
                                            key=selectbox_key)

        df = dataframe.groupby('created_at')
        sentiment_mean = df['sentiment'].mean().reset_index()
        tweet_count = df['sentiment'].size().reset_index()
        tweet_count = tweet_count.rename(columns={'sentiment': 'tweets'})
        grouped_df = sentiment_mean.join(tweet_count.set_index('created_at'), on='created_at', how='left')

        min_date, max_date = grouped_df['created_at'].min(), grouped_df['created_at'].max()
        datelist = [entry.strftime("%Y-%m-%d") for entry in pd.date_range(min_date, max_date)]
        new_dataframe = pd.DataFrame.from_dict({'created_at': datelist})
        joined_grouped = new_dataframe.join(grouped_df.set_index('created_at'), on='created_at', how='outer')
        joined_grouped.fillna(value=0, inplace=True)
        joined_grouped['sentiment_times_tweets'] = joined_grouped['sentiment'] * joined_grouped['tweets']

        joined_grouped['Week_Number'] = pd.to_datetime(joined_grouped['created_at'], errors='coerce').dt.strftime('%U')
        joined_grouped['Month'] = pd.to_datetime(joined_grouped['created_at'], errors='coerce').dt.strftime('%m')
        joined_grouped['Year'] = pd.to_datetime(joined_grouped['created_at'], errors='coerce').dt.strftime('%Y')

        if time_step_select_box == 'Day':
            key = 'created_at'
        elif time_step_select_box == 'Week':
            key = 'Year - Week'
            joined_grouped[key] = joined_grouped['Year'] + " - " + joined_grouped['Week_Number']
        elif time_step_select_box == 'Month':
            key = 'Year - Month'
            joined_grouped[key] = joined_grouped['Year'] + " - " + joined_grouped['Month']
        elif time_step_select_box == 'Year':
            key = 'Year'

        return joined_grouped.groupby(key), key

    def get_grouped_by_data(dataframe, selectbox_key):
        grouped_sentiment, key = get_group_by_dataframe(dataframe, selectbox_key)
        grouped_dataframe: pd.DataFrame = (grouped_sentiment['sentiment_times_tweets'].sum() / grouped_sentiment[
            'tweets'].sum()).reset_index()
        grouped_dataframe.rename(columns={0: 'sentiment'}, inplace=True)
        grouped_dataframe['tweets'] = grouped_sentiment['tweets'].sum().reset_index()['tweets']
        grouped_dataframe = grouped_dataframe.sort_values(key)
        return grouped_dataframe, key

    def sentiment_plot(dataframe, selectbox_key):
        grouped_sentiment, key_sentiment = get_grouped_by_data(dataframe, selectbox_key)
        st.altair_chart(alt.Chart(grouped_sentiment).mark_line(point=True).encode(
            x=key_sentiment,
            y='sentiment'
        ), use_container_width=True)

    sentiment_plot(tweets_df.copy(), 'grouped_sentiment')

    st.subheader("Normalized Sentiment Data Over Time")
    st.text('Here the sentiment values are normalized to 0 mean and std 1 before computing the group-wise average.')
    dataframe = tweets_df.copy()
    mean, std = dataframe['sentiment'].mean(), dataframe['sentiment'].std()
    dataframe['sentiment'] = (dataframe['sentiment'] - mean) / std
    sentiment_plot(dataframe, 'grouped_normalized_sentiment')

    st.subheader("Number of Tweets Over Time")
    dataframe = tweets_df.copy()
    grouped_number_of_tweets, key_number_of_tweets = get_grouped_by_data(dataframe,
                                                                            'grouped_number_of_tweets'
                                                                            )
    grouped_number_of_tweets = grouped_number_of_tweets[[key_number_of_tweets, 'tweets']]
    st.altair_chart(alt.Chart(grouped_number_of_tweets).mark_line(point=True).encode(
        x=key_number_of_tweets,
        y='tweets'
    ), use_container_width=True)

    st.subheader("Text Input")
    st.text("Here you can input text and the sentiment will be returned.")

    with st.form("text_input_form"):
        text_input = st.text_input("Text")
        submit_text_input = st.form_submit_button("Submit")
    sentiment_output = st.text("", )
    if submit_text_input:
        # request from backend
        url = f"{BACKEND_URL}:{BACKEND_PORT}/{TEXT_SENTIMENT}"
        obj = {'text': str(text_input)}

        request_answer = requests.post(url, json=obj)
        request_answer = request_answer.json()
        sentiment = request_answer['sentiment']
        sentiment_output.text(sentiment)


if __name__ == '__main__':
    init_frontend()
