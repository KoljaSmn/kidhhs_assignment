import os

import pandas as pd
from kidhhs.config.config import COVID_TWITTER_DATASET_DIR
import logging


def load_covid_twitter_dataset():

    full_dataframe = None

    logging.info('Loading Covid Dataset.')

    for file in os.listdir(COVID_TWITTER_DATASET_DIR):
        filename = os.fsdecode(file)
        if filename.endswith('csv'):
            dataframe = pd.read_csv(os.path.join(COVID_TWITTER_DATASET_DIR, filename))


            if full_dataframe is None:
                full_dataframe = dataframe
            else:
                full_dataframe = pd.concat([full_dataframe, dataframe])


    # we remove some columns from the dataset, especially labeled sentiments
    full_dataframe = full_dataframe.drop(['id', 'source', 'hashtags', 'user_mentions', 'compound', 'neg',
                                          'neu', 'pos', 'sentiment'], axis=1)

    return full_dataframe
