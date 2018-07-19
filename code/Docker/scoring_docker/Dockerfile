### docker run  -v $(pwd)/code/code/results:/usr/local/src/NAB/results  -v $(pwd)/code/code/data:/usr/local/src/NAB/data -it abdulrehmanliaqat/scoring_docker bash

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
ADD /NAB ./NAB

# Change directory to NAB
WORKDIR /usr/local/src/NAB/

VOLUME /usr/local/src/NAB/

ADD requirements.txt ./

RUN pip install -r requirements.txt


