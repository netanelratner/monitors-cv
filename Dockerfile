FROM python:3.6-buster
RUN apt-get update && apt-get install -yyq  libzbar0 libjpeg-dev && apt-get clean
ADD requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt
ADD . /opt/app
RUN pip install -e /opt/app
RUN pytest /opt/app
CMD cvmonitor