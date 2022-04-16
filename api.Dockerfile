FROM python:3.8

COPY ./src /src
COPY ./api /api
COPY ./requirements_api.txt /requirements_api.txt

RUN pip install -r requirements_api.txt

EXPOSE 8085


CMD ["uvicorn", "api.api:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8085"]
