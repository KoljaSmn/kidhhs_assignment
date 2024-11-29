import pandas as pd
from flask import Flask, jsonify, request, Response

from kidhhs.config.config import BACKEND_PORT, TWEETS_AND_SENTIMENT_POST_NAME, SENTIMENT_POST_NAME, TEXT_SENTIMENT
from kidhhs.backend.sqlite import get_tweets_and_sentiment_from_to, get_sentiment_from_to
from kidhhs.backend.sentiment_analysis import sentiment_for_text

# flask app to run the backend api
app = Flask(__name__)

@app.route("/")
def hello_world():
    """
    Method to check whether the backend api is running and reachable.
    :return:
    """
    return "Hello, World!"


@app.route(f'/{TWEETS_AND_SENTIMENT_POST_NAME}', methods=['POST'])
def get_tweets_and_sentiment():
    """
    Returns joined tweet and sentiment data for tweets between date_from and date_to.
    Returned data contains id, the date of the tweet of the sentiment the tweet and the sentiment score.

    Expects json with two entries 'date_from' and 'date_to'.

    :return:
    """
    request_data = request.get_json()
    date_from = request_data['date_from']
    date_to = request_data['date_to']
    dataframe = get_tweets_and_sentiment_from_to(date_from, date_to)
    dataframe = dataframe[['created_at', 'original_text', 'sentiment']]
    return Response(dataframe.to_json(orient="records"), mimetype='application/json')


@app.route(f'/{SENTIMENT_POST_NAME}', methods=['POST'])
def get_sentiments():
    """
    Returns sentiment data for tweets between date_from and date_to.
    Returned data contains id, the date of the tweet of the sentiment and the sentiment score.

    Expects json with two entries 'date_from' and 'date_to'.

    :return:
    """
    request_data = request.get_json()
    date_from = request_data['date_from']
    date_to = request_data['date_to']
    dataframe = get_sentiment_from_to(date_from, date_to)
    return Response(dataframe.to_json(orient="records"), mimetype='application/json')


@app.route(f'/{TEXT_SENTIMENT}', methods=['POST'])
def get_sentiment_for_text():
    """
    Returns the sentiment score for the given text.

    Expects json with one entry 'text'.

    :return:
    """
    request_data = request.get_json()
    response_data = {'sentiment': str(sentiment_for_text(request_data['text']))}
    return jsonify(response_data)


def run_app():
    app.run(host='0.0.0.0', port=BACKEND_PORT, debug=False, use_reloader=False)
