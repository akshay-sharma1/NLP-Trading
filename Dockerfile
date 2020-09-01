FROM python:latest

WORKDIR /Trump_Tweets

# copy all requirements to docker image
COPY requirements.txt .
RUN pip install -r requirements.txt

# add necessary nltk packages
RUN python -c "import nltk; nltk.download('punkt')"
RUN python -c "import nltk; nltk.download('stopwords')"
RUN python -c "import nltk; nltk.download('wordnet')"

ADD *.py ./

CMD ["python", "bot.py"]

