import streamlit as st
from datetime import datetime
from collections import namedtuple
import altair as alt
import math
import pandas as pd
import requests

import sys
import os

sys.path += ['.']

from kidhhs.config.config import BACKEND_PORT, SENTIMENT_POST_NAME, TEXT_SENTIMENT, TWEETS_AND_SENTIMENT_POST_NAME, BACKEND_URL


@st.cache_data
def _load_tweet_data(date_from, date_to):
    # load data for showing tweets
    url = f"{BACKEND_URL}:{BACKEND_PORT}/{TWEETS_AND_SENTIMENT_POST_NAME}"
    obj = {'date_from': date_from, 'date_to': date_to}
    request_answer = requests.post(url, json=obj).json()
    tweets_df = pd.DataFrame.from_dict(request_answer)
    return tweets_df


def init_frontend():
    st.set_page_config(layout="wide")

    st.header("Covid Twitter Sentiment Analysis Over Time")

    st.subheader('Date Selection')
    st.text(
        'Here you can choose the date range to consider tweets from. The selection will be used in all following subsections.')
    tweets_date_col1, tweets_date_col2 = st.columns(2)

    date_input_count = 0

    with tweets_date_col1:
        from_date_input_tweets = st.date_input('From date', value=datetime(2019, 12, 1), key=date_input_count)
        date_input_count += 1
    with tweets_date_col2:
        to_date_input_tweets = st.date_input('To date', value=datetime(2021, 12, 31), key=date_input_count)
        date_input_count += 1

    with st.form(key='submit_form'):
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        tweets_df = _load_tweet_data(str(from_date_input_tweets), str(to_date_input_tweets))

        st.subheader("Tweets")

        with st.expander("Tweet Sentiments"):
            st.dataframe(tweets_df, use_container_width=True)

        # analysis of sentiments over time
        st.subheader("Sentiment Data Over Time")

        def get_group_by_dataframe(dataframe, selectbox_key):
            time_step_select_box = st.selectbox('Figure Group By', ['Day', 'Week', 'Month', 'Year'], index=2,
                                                key=selectbox_key)
            dataframe['Week_Number'] = pd.to_datetime(dataframe['created_at'], errors='coerce').dt.strftime('%U')
            dataframe['Month'] = pd.to_datetime(dataframe['created_at'], errors='coerce').dt.strftime('%m')
            dataframe['Year'] = pd.to_datetime(dataframe['created_at'], errors='coerce').dt.strftime('%Y')

            if time_step_select_box == 'Day':
                key = 'created_at'
            elif time_step_select_box == 'Week':
                key = 'Year - Week'
                dataframe[key] = dataframe['Year'] + " - " + dataframe['Week_Number']
            elif time_step_select_box == 'Month':
                key = 'Year - Month'
                dataframe[key] = dataframe['Year'] + " - " + dataframe['Month']
            elif time_step_select_box == 'Year':
                key = 'Year'

            return dataframe.groupby(key)['sentiment'], key


        def sentiment_plot(dataframe, selectbox_key):
            grouped_sentiment, key_sentiment = get_group_by_dataframe(dataframe, selectbox_key)
            grouped_sentiment = grouped_sentiment.mean().reset_index()
            grouped_sentiment = grouped_sentiment.sort_values(key_sentiment)
            st.altair_chart(alt.Chart(grouped_sentiment).mark_line(point=True).encode(
                x=key_sentiment,
                y='sentiment'
            ), use_container_width=True)

        sentiment_plot(tweets_df.copy(), 'grouped_sentiment')

        st.subheader("Normalized Sentiment Data Over Time")
        st.text('Here the sentiment values are normalized to 0 mean and std 1 before computing the group-wise average.')
        dataframe = tweets_df.copy()
        mean, std = dataframe['sentiment'].mean(), dataframe['sentiment'].std()
        dataframe['sentiment'] = (dataframe['sentiment'] - mean)/std
        sentiment_plot(dataframe, 'grouped_normalized_sentiment')

        st.subheader("Number of Tweets Over Time")
        dataframe = tweets_df.copy()
        grouped_number_of_tweets, key_number_of_tweets = get_group_by_dataframe(dataframe, 'grouped_number_of_tweets')
        grouped_number_of_tweets = grouped_number_of_tweets.size().reset_index()
        grouped_number_of_tweets = grouped_number_of_tweets.sort_values(key_number_of_tweets)
        st.altair_chart(alt.Chart(grouped_number_of_tweets).mark_line(point=True).encode(
            x=key_number_of_tweets,
            y='sentiment'
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
