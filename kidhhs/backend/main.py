import kidhhs.backend.data_initialization
import kidhhs.backend.api

import torch
import logging
import threading

logging.getLogger().setLevel(logging.INFO)


def init_backend():
    logging.info(f'cuda available: {torch.cuda.is_available()}')

    threading.Thread(target=kidhhs.backend.api.run_app).start()
    kidhhs.backend.data_initialization.init_database()


if __name__ == '__main__':
    init_backend()
