# To build: docker build -t pixarninja/tf-gpu .
# To run:   winpty docker run -it --rm --name tf-gpu pixarninja/tf-gpu bash
FROM continuumio/anaconda3

RUN /opt/conda/bin/conda install python
RUN /opt/conda/bin/conda install pip
RUN /opt/conda/bin/conda install tensorflow-gpu==1.13.1
RUN /opt/conda/bin/pip install blocksparse

COPY ./app /app
