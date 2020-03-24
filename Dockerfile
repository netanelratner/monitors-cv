FROM nvidia/cuda:10.1-cudnn7-runtime
RUN apt-get update && apt-get install -yy  libzbar0 libjpeg-dev python3-pip python3-venv &&  rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -mvenv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ADD requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt --no-cache-dir
ADD . /opt/app
RUN pip install -e /opt/app --no-cache-dir
RUN pytest /opt/app
CMD cvmonitor