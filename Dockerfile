FROM python:3.9-bullseye

ADD requirements.txt /tmp/

RUN pip install -r /tmp/requirements.txt

RUN python --version && pip list 

ADD code /code
WORKDIR /code