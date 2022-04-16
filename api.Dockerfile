FROM python:3.8

COPY ./src /src
COPY ./api /api
COPY ./requirements_api.txt /requirements_api.txt

RUN pip install -r requirements_api.txt

EXPOSE 8000

CMD ["uvicorn","api.api:app","--port","8000"]