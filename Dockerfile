FROM python:3.11-slim

RUN apt-get update
RUN apt-get install -y bzip2 git lrzsz wget vim git-lfs
RUN apt-get install -y g++

WORKDIR /data/tianxing/PycharmProjects/OpenUnsloth

COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --upgrade -r /data/tianxing/PycharmProjects/OpenUnsloth/requirements.txt
