FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /home/kidhhs_assignment/
WORKDIR /home/kidhhs_assignment/

RUN pip3 install -r docker/backend/requirements.txt

ENV PYTHONPATH=/home/kidhhs_assignment
ENTRYPOINT ["python", "kidhhs/backend/main.py"]