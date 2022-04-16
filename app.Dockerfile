FROM python:3.8

WORKDIR /application

COPY ./app /application/app
COPY ./requirements/requirements_app.txt /application/requirements_app.txt

RUN pip install -r requirements_app.txt

ENV PYTHONPATH "/application/"

EXPOSE 8501

CMD ["streamlit","run","app/app.py"]