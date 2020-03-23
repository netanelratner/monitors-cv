FROM python:3.6.10-buster
RUN apt-get update && apt-get install -yyq  libzbar0 && apt-get clean
ADD . /opt/app
RUN pip install -e /opt/app
RUN start-server