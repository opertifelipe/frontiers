FROM python:3.8

COPY ./src /src
COPY ./app /app
COPY ./requirements.txt /requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit","run","app/app.py"]