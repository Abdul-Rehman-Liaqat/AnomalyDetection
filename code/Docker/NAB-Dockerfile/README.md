# README #

This README explains how to get NAB up and running on your machine. NAB is a novel benchmark for evaluating algorithms for anomaly detection in streaming,  real-time applications. It is comprised of over 50 labeled real-world and artificial timeseries data files plus a novel scoring mechanism designed for real-time applications.

In order to install NAB on you local machine you can use NAB docker file provided here. It's been built on top of another docker from "numenta/nupic".  To run a docker file you need to first, install docker on you machine and then build the current file so it is ready to run. 

To install docker you need to follow instruction on: 
```sh
https://docs.docker.com/engine/installation/
```
When done by installing docker, you need to buil this docker using:
```sh
docker build -t you_docker_name .
```
Now you can run the docker and NAB will be there for you:
```sh
docker run your_docker_name
```
You can find full explanation about NAB and how to install it on:
```sh
https://github.com/numenta/NAB
```
But to have a quick start and see some results you can run the following command:
```sh
cd /path/to/nab
python run.py -d numenta --detect --score --normalize
```
Good Luck
