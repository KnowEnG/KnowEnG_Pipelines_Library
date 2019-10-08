FROM ubuntu:bionic-20190912.1
LABEL Xi Chen="xichen24@illinois.edu" \
      Jing Ge="jingge2@illinois.edu" \
      Dan Lanier="lanier4@illinois.edu" \
      Nahil Sobh="sobh@illinois.edu"

# Install packages and their dependencies
RUN apt-get update && \
    apt-get install -y vim python3-pip git libblas-dev liblapack-dev libatlas-base-dev gfortran libfreetype6-dev libxft-dev curl

RUN pip3 install -I numpy pandas scipy==0.19.1 scikit-learn==0.19.2 pyyaml xmlrunner



RUN pip3 install -I pandas==0.25.1
RUN pip3 install -I scipy==1.3.1 scikit-learn==0.21.3 pyyaml==5.1.2 xmlrunner==1.7.7
RUN pip3 install -I knpackage==0.1.26 redis==3.3.8
