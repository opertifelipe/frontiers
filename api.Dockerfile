FROM python:3.8

COPY ./src /src
COPY ./api /api
COPY ./requirements.txt /requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn","api.api:app","--port 8000"]