### docker run comman is : docker run  -v /home/abdulliaqat/Desktop/thesis/AnomalyDetection/code/rough/results:/usr/local/src/NAB/results -it scoring_docker bash

## Set the base image to Ubuntu
FROM numenta/nupic

# File Author / Maintainer
LABEL MAINTAINER="Abdul Rehman Liaqat <liaqat@uni-hildesheim.de>"

# Set enviroment variables needed by NuPIC
ENV NUPIC /usr/local/src/NAB

# OPF needs this
ENV USER docker

# Change work directory
WORKDIR /usr/local/src

# Clone NAB repo
ADD /NAB ./

# Change directory to NAB
WORKDIR /usr/local/src/NAB/

VOLUME /usr/local/src/NAB/

RUN pip install -r requirements.txt


