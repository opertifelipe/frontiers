FROM python:3.8

ARG DEVICE

COPY ./src /src
COPY ./requirements.txt /requirements.txt

RUN pip install -r requirements.txt

CMD "python src/main_train.py ${DEVICE}"