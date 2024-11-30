import os
import json

# the data directory
DATA_DIR = 'kidhhs/data/'
# directory for the covid dataset csv files
COVID_TWITTER_DATASET_DIR = os.path.join(DATA_DIR, 'covid_twitter_dataset')
# path to the sqlite database
DATABASE_PATH = os.path.join(DATA_DIR, 'sqlite.db')
# path to the config.json which saves information, e.g. whether the database has been initialized already
CONFIG_JSON = 'kidhhs/config/config.json'
# defines the batch size for the llm
LLM_BATCH_SIZE = 100

N_ENTRIES_DATABASE = None # can be used to reduce the time for initialization, None for all entries

BACKEND_URL = 'http://backend' # 'http://backend' # use 'http://backend' when using docker compose and 'http://localhost' # when running both servers locally
# the port to reach the backend
BACKEND_PORT = 8080

TWEETS_AND_SENTIMENT_POST_NAME = 'tweets'
SENTIMENT_POST_NAME = 'sentiment'
TEXT_SENTIMENT = 'textsentiment'
DB_UPDATE = 'dpupdate'


def __load_config_json():
    """
    loads the config json
    :return:
    """
    if not os.path.isfile(CONFIG_JSON):
        __save_config_json({})


    with open(CONFIG_JSON, 'r') as file:
        data = json.load(file)
    return data


def __save_config_json(data):
    """
    Saves the given data to the config json
    :param data:
    :return:
    """
    with open(CONFIG_JSON, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def is_database_initialized():
    """
    Returns whether the database has been initialized
    :return:
    """
    data = __load_config_json()
    # check the value set in the config json
    initialized_set = data['db_initialized'] if 'db_initialized' in data else False
    # iff the value in the config json is set and the database file exists, return True
    return initialized_set and os.path.isfile(DATABASE_PATH)


def set_database_initialized():
    """
    set the value in the config json corresponding to whether the database has been initialized
    :return:
    """
    data = __load_config_json()
    data['db_initialized'] = True
    __save_config_json(data)

