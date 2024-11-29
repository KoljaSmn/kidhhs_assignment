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

        # load data for showing tweets
        url = f"{BACKEND_URL}:{BACKEND_PORT}/{TWEETS_AND_SENTIMENT_POST_NAME}"
        obj = {'date_from': str(from_date_input_tweets), 'date_to': str(to_date_input_tweets)}
        request_answer = requests.post(url, json=obj).json()
        tweets_df = pd.DataFrame.from_dict(request_answer)

        st.subheader("Tweets")

        with st.expander("Tweet Sentiments"):
            st.dataframe(tweets_df, use_container_width=True)

        # analysis of sentiments over time
        st.subheader("Sentiment Data Over Time")



        def get_group_by_dataframe(selectbox_key):
            time_step_select_box = st.selectbox('Figure Group By', ['Day', 'Week', 'Month', 'Year'], index=2,
                                                key=selectbox_key)
            tweets_df['Week_Number'] = pd.to_datetime(tweets_df['created_at'], errors='coerce').dt.strftime('%U')
            tweets_df['Month'] = pd.to_datetime(tweets_df['created_at'], errors='coerce').dt.strftime('%m')
            tweets_df['Year'] = pd.to_datetime(tweets_df['created_at'], errors='coerce').dt.strftime('%Y')

            if time_step_select_box == 'Day':
                key = 'created_at'
            elif time_step_select_box == 'Week':
                key = 'Year - Week'
                tweets_df[key] = tweets_df['Year'] + " - " + tweets_df['Week_Number']
            elif time_step_select_box == 'Month':
                key = 'Year - Month'
                tweets_df[key] = tweets_df['Year'] + " - " + tweets_df['Month']
            elif time_step_select_box == 'Year':
                key = 'Year'

            return tweets_df.groupby(key)['sentiment'], key

        grouped_sentiment, key_sentiment = get_group_by_dataframe('grouped_sentiment')
        grouped_sentiment = grouped_sentiment.mean().reset_index()
        grouped_sentiment = grouped_sentiment.sort_values(key_sentiment)
        st.altair_chart(alt.Chart(grouped_sentiment).mark_circle().encode(
            x=key_sentiment,
            y='sentiment'
        ), use_container_width=True)

        st.subheader("Number of Tweets Over Time")
        grouped_number_of_tweets, key_number_of_tweets = get_group_by_dataframe('grouped_number_of_tweets')
        grouped_number_of_tweets = grouped_number_of_tweets.size().reset_index()
        grouped_number_of_tweets = grouped_number_of_tweets.sort_values(key_number_of_tweets)
        st.altair_chart(alt.Chart(grouped_number_of_tweets).mark_circle().encode(
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
