# To build: docker build -t pixarninja/tf-gpu .
# To run:   winpty docker run -it --rm --name tf-gpu pixarninja/tf-gpu bash
#       :   docker run -it -p 1234:8888 --name tf-gpu pixarninja/tf-gpu:jupyter
#       :   docker run --rm --runtime=nvidia -it pixarninja/tf-gpu
#FROM tensorflow/tensorflow:nightly-gpu-py3-jupyter
FROM tensorflow/tensorflow:1.13.1-gpu-py3

RUN python -m pip install --upgrade pip
RUN pip install blocksparse

COPY ./app /app

RUN ./app/blocksparse_setup.sh
