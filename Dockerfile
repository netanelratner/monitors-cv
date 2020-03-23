FROM ubuntu:18.04
RUN apt-get update && apt-get install -yyq  libzbar0 && apt-get clean
ADD . /opt/app
RUN pip install -e /opt/app
RUN start-server