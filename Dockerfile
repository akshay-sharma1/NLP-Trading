FROM python:latest

WORKDIR /Trump_Tweets

# copy all requirements and cache them
COPY requirements.txt .
RUN pip install -r requirements.txt

ADD *.py ./

CMD ["python", "bot.py"]

