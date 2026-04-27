FROM python:3.11-slim

WORKDIR /app

COPY flask_app/ /app/

COPY models/vectorizer.pkl /app/models/vectorizer.pkl
COPY models/model.pkl /app/models/model.pkl

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

#local
#CMD ["python", "app.py"]

#Prod
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]