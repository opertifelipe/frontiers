FROM python:3.8

COPY ./src /src
COPY ./app /app
COPY ./requirements_app.txt /requirements_app.txt

RUN pip install -r requirements_app.txt

EXPOSE 8501

CMD ["streamlit","run","app/app.py"]