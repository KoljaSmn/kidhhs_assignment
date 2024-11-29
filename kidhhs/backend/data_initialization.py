import logging
import os

import pandas as pd
from tqdm_loggable.auto import tqdm

from kidhhs.backend.dataset_loader import load_covid_twitter_dataset
from kidhhs.backend.sqlite import write_covid_twitter_dataframe, setup_database
from kidhhs.config.config import is_database_initialized, set_database_initialized, LLM_BATCH_SIZE, N_ENTRIES_DATABASE
from kidhhs.backend.sentiment_analysis import sentiment_for_text_batch


def init_database():
    """
    Function to initialize the dataset.
    :return:
    """
    if not is_database_initialized():
        # if the database has not been initialized already
        # set it up, generate table
        setup_database()

        logging.info('Database not initialized yet.')

        # load the csv files to memory
        dataframe = load_covid_twitter_dataset()
        # add an id column and use it as index
        dataframe['id'] = list(range(dataframe.shape[0]))
        dataframe.set_index('id', inplace=True)


        # variables to process batches
        batch_tweets, batch_ids = [], []

        def _init_batch(tweet_batch, id_batch):
            """
            Computes sentiments foe a batch and saves both tweet data and sentiments to the databes
            :param tweet_batch:
            :param id_batch:
            :return:
            """
            sentiment_scores = sentiment_for_text_batch(tweet_batch)
            # to save to the database, we use all the information, including author and original_text
            # uses the ids to retrieve information from the full dataframe
            dataframe_batch = dataframe.loc[id_batch]
            sentiment_dataframe = pd.DataFrame.from_dict({'id': id_batch, 'sentiment': sentiment_scores})
            sentiment_dataframe.set_index('id', inplace=True)
            write_covid_twitter_dataframe(dataframe_batch, sentiment_dataframe)


        # iterate through the dataframe rows and collect data for the batch
        tqdm_total = min(dataframe.shape[0], N_ENTRIES_DATABASE) if N_ENTRIES_DATABASE else dataframe.shape[0]
        for index, row in tqdm(dataframe.iterrows(), total=tqdm_total):
            # append to the current batch
            batch_tweets.append(row['clean_tweet'])
            batch_ids.append(index)
            if len(batch_tweets) == LLM_BATCH_SIZE:
                # if the batch_size is reached, process the current batch
                _init_batch(batch_tweets, batch_ids)
                batch_tweets, batch_ids = [], []

            if N_ENTRIES_DATABASE and index==N_ENTRIES_DATABASE:
                break

        if batch_tweets:
            # process last, incomplete batch
            _init_batch(batch_tweets, batch_ids)

        set_database_initialized()
    else:
        logging.info('Database already initialized.')

