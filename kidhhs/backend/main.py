import kidhhs.backend.data_initialization
import kidhhs.backend.api

import torch
import logging
import threading

logging.getLogger().setLevel(logging.INFO)


def init_backend():
    """
    Initializes the backend by setting up the database and flask api.
    :return:
    """
    logging.info(f'cuda available: {torch.cuda.is_available()}')

    # run the flask app in a separate thread
    threading.Thread(target=kidhhs.backend.api.run_app).start()
    # such that the database can be prepared while the api is already running
    # thereby, parts of the data can already be used while the database is being created
    kidhhs.backend.data_initialization.init_database()


if __name__ == '__main__':
    init_backend()
