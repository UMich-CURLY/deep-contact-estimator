FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

LABEL version="0.5"

# FROM ros:melodic

USER root

RUN apt-get update && apt-get install -y vim

RUN pip install scipy

RUN pip install tensorboard
