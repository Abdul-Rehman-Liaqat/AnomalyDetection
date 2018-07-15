## docker run --user=root -p 8888:8888 -v /home/abdulliaqat/Desktop/thesis/AnomalyDetection/code/rough:/executor/execution_code execution_docker

## Set the base image to Ubuntu
FROM jupyter/tensorflow-notebook

# File Author / Maintainer
LABEl MAINTAINER="Abdul Rehman Liaqat <liaqat@uni-hildesheim.de>"

WORKDIR /executor

ADD requirements.txt /executor

RUN pip install -r requirements.txt

VOLUME /executor/execution_code

CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888","--no-browser","--allow-root"]    

