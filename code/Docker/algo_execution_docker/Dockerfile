## Set the base image to Ubuntu
FROM jupyter/tensorflow-notebook

# File Author / Maintainer
LABEl MAINTAINER="Abdul Rehman Liaqat <liaqat@uni-hildesheim.de>"

WORKDIR /executor

ADD requirements.txt /executor

RUN pip install -r requirements.txt

CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888","--no-browser","--allow-root"]
    

