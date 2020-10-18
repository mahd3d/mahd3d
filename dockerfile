FROM python:3.8-slim

RUN mkdir app
WORKDIR app
RUN touch is_docker_container

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="$PYTHONPATH:./"

RUN apt update && apt install -y libxerces-c-dev build-essential

COPY ./requirements.txt ./requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .

RUN which python3 && python3 --version
RUN which pip3 && pip3 --version
RUN pip3 freeze

ENTRYPOINT python3 main.py
