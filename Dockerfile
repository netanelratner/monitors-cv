FROM nvidia/cuda:10.1-cudnn7-runtime
ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get update && apt-get install -yy  libzbar0 libjpeg-turbo8-dev libz-dev python3-pip python3-venv git python3-tk &&  rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -mvenv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ADD requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt --no-cache-dir
COPY . /opt/app
COPY .git /opt/app/.git
WORKDIR /opt/app/
RUN pip install -e /opt/app --no-cache-dir
RUN pytest /opt/app
RUN cvmonitor-get-models
CMD cvmonitor