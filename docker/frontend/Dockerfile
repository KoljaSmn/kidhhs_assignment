FROM python:3.11.10

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /home/kidhhs_assignment/
WORKDIR /home/kidhhs_assignment/

RUN pip3 install -r docker/frontend/requirements.txt

ENV PYTHONPATH=/home/kidhhs_assignment
ENTRYPOINT ["streamlit", "run", "kidhhs/frontend/app.py", "--server.port=8081", "--server.address=0.0.0.0"]