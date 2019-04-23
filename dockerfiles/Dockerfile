# Start with cuDNN base image
FROM floydhub/torch:7-gpu-py3.6
MAINTAINER Cong Feng <cgsdfc@126.com>

RUN pip install numpy

RUN luarocks install torchx
RUN luarocks install ids
RUN luarocks install logroll
