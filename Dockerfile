FROM ubuntu:20.04

# setup packages
RUN apt-get update -y
RUN apt-get install -y python3 python-is-python3 python3-pip python3-venv

# copy local files
RUN mkdir /home/TFB
COPY . /home/TFB

# create virtualenv, activate and upgrade the env-oriented pip
RUN python -m venv /env
ENV PATH="/env/bin:$PATH"
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r /home/TFB/requirements-docker.txt