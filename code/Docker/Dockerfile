## Set the base image to Ubuntu
FROM numenta/nupic

# File Author / Maintainer
MAINTAINER Abdul Rehman Liaqat (liaqat@uni-hildesheim.de)

# Set enviroment variables needed by NuPIC
ENV NUPIC /usr/local/src/NAB

# OPF needs this
ENV USER docker

# Change work directory
WORKDIR /usr/local/src

# Clone NAB repo
RUN git clone https://github.com/numenta/NAB

# Change directory to NAB
WORKDIR /usr/local/src/NAB/

# Install NAB python requirments
RUN pip install pandas
RUN pip install simplejson
RUN pip install mlflow

# Install NAB
RUN python setup.py install --user

WORKDIR /home/docker
