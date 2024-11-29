import logging
import os

import pandas as pd
from tqdm_loggable.auto import tqdm

from kidhhs.backend.dataset_loader import load_covid_twitter_dataset
from kidhhs.backend.sqlite import write_covid_twitter_dataframe, setup_database
from kidhhs.config.config import is_database_initialized, set_database_initialized, LLM_BATCH_SIZE, N_ENTRIES_DATABASE
from kidhhs.backend.sentiment_analysis import sentiment_for_text_batch


def init_database():
    if not is_database_initialized():
        setup_database()
        logging.info('Database not initialized yet.')
        dataframe = load_covid_twitter_dataset()
        dataframe['id'] = list(range(dataframe.shape[0]))
        dataframe.set_index('id', inplace=True)

        batch_tweets, batch_ids = [], []

        def _init_batch(tweet_batch, id_batch):
            sentiment_scores = sentiment_for_text_batch(tweet_batch)
            dataframe_batch = dataframe.loc[id_batch]
            sentiment_dataframe = pd.DataFrame.from_dict({'id': id_batch, 'sentiment': sentiment_scores})
            sentiment_dataframe.set_index('id', inplace=True)
            write_covid_twitter_dataframe(dataframe_batch, sentiment_dataframe)


        tqdm_total = min(dataframe.shape[0], N_ENTRIES_DATABASE) if N_ENTRIES_DATABASE else dataframe.shape[0]
        for index, row in tqdm(dataframe.iterrows(), total=tqdm_total):
            batch_tweets.append(row['clean_tweet'])
            batch_ids.append(index)
            if len(batch_tweets) == LLM_BATCH_SIZE:
                _init_batch(batch_tweets, batch_ids)
                batch_tweets, batch_ids = [], []


            if N_ENTRIES_DATABASE and index==N_ENTRIES_DATABASE:
                break

        if batch_tweets:
            _init_batch(batch_tweets, batch_ids)

        set_database_initialized()
    else:
        logging.info('Database already initialized.')

