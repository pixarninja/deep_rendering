# To build: docker build -f Dockerfile.90_3 -t pixarninja/sparse .
# To run:   docker run --rm --runtime=nvidia -it --name sparse -v ~/Git/dynamic_frame_generator/python/training:/app/training:ro -v ~/Git/dynamic_frame_generator/python/sparse_attention/output:/app/output pixarninja/sparse
#FROM tensorflow/tensorflow:nightly-gpu-py3-jupyter
FROM pixarninja/tf-gpu

# Copy application and configure blocksparse
COPY ./app /app
WORKDIR /app 

# Run the python file as a command on startup
CMD python attention.py --alpha 0.90 --beta 3 --blockDim 32 --inType frame ;\
    python attention.py --alpha 0.90 --beta 3 --blockDim 32 --inType cifar ;\
    python attention.py --alpha 0.90 --beta 7 --blockDim 32 --inType frame ;\
    python attention.py --alpha 0.90 --beta 7 --blockDim 32 --inType cifar ;\
    python attention.py --alpha 0.75 --beta 7 --blockDim 32 --inType frame ;\
    python attention.py --alpha 0.75 --beta 7 --blockDim 32 --inType cifar ;\
    python attention.py --useRange --blockDim 32 --inType frame ;\
    python attention.py --useRange --blockDim 32 --inType cifar
