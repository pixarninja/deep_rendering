#!/bin/bash

docker build -t pixarninja/sparse .
docker run --rm --runtime=nvidia -it --name sparse \
       -v ~/Git/dynamic_frame_generator/python/training:/app/training:ro \
       -v ~/Git/dynamic_frame_generator/python/sparse_attention/output:/app/output \
       pixarninja/sparse
