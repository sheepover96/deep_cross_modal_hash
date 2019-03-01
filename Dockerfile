FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends build-essential \
    python3-dev python3-pip python3-setuptools

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY ./ ./
RUN pip3 install --no-cache-dir -r requirements.txt
RUN python3 -m spacy download en

CMD python3 main.py