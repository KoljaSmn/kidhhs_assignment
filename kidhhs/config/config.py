import os
import json

DATA_DIR = 'kidhhs/data/'
COVID_TWITTER_DATASET_DIR = os.path.join(DATA_DIR, 'covid_twitter_dataset')
DATABASE_PATH = os.path.join(DATA_DIR, 'sqlite.db')
CONFIG_JSON = 'kidhhs/config/config.json'
LLM_BATCH_SIZE = 100
N_ENTRIES_DATABASE = None # can be used to reduce the time for initialization, None for all entries
BACKEND_URL = 'http://backend' # when using docker compose http://localhost' # when running both servers locally
BACKEND_PORT = 8080
TWEETS_AND_SENTIMENT_POST_NAME = 'tweets'
SENTIMENT_POST_NAME = 'sentiment'
TEXT_SENTIMENT = 'textsentiment'


def __load_config_json():
    if not os.path.isfile(CONFIG_JSON):
        __save_config_json({})


    with open(CONFIG_JSON, 'r') as file:
        data = json.load(file)
    return data


def __save_config_json(data):
    with open(CONFIG_JSON, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def is_database_initialized():
    data = __load_config_json()
    initialized_set = data['db_initialized'] if 'db_initialized' in data else False
    return initialized_set and os.path.isfile(DATABASE_PATH)


def set_database_initialized():
    data = __load_config_json()
    data['db_initialized'] = True
    __save_config_json(data)

