FROM python:3.8

WORKDIR /app 

COPY ./src /app/src
COPY ./requirements_train.txt /app/requirements_train.txt
ENV PYTHONPATH "/app/"
RUN pip install -r requirements_train.txt

CMD ["python","src/main_train.py"]