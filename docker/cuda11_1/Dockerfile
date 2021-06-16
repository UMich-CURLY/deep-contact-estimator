FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

LABEL version="0.5"

USER root

# scipy, tensorboard
RUN pip install scipy
RUN pip install tensorboard
RUN pip install -U scikit-learn

# build essential, cmake, vim, git
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install cmake
RUN apt-get update && apt-get install -y vim
RUN apt-get install -y build-essential
RUN apt-get update && apt-get install -y git-all

# lcm 
RUN apt-get install -y libglib2.0-dev
RUN mkdir -p /home/root/tmp/
RUN cd /home/root/tmp/ \
    && git clone https://github.com/lcm-proj/lcm.git
RUN cd /home/root/tmp/lcm/ \
    && mkdir build \
    && cd build \ 
    && cmake .. -DLCM_ENABLE_PYTHON=ON && make -j && make install
RUN cd /home/root/tmp/lcm/lcm-python/\
    && python setup.py install
