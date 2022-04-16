FROM python:3.8

WORKDIR /app

COPY ./api /app/api
COPY ./requirements/requirements_api.txt /app/requirements_api.txt

RUN pip install -r requirements_api.txt
ENV PYTHONPATH "/app/"

EXPOSE 8000

CMD ["uvicorn", "api.api:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]
