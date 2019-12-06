# To build: docker build -t pixarninja/tf-gpu .
# To run:   winpty docker run -it --rm --name tf-gpu pixarninja/tf-gpu bash
#       :   docker run -it -p 1234:8888 --name tf-gpu pixarninja/tf-gpu:jupyter
#       :   docker run --rm --runtime=nvidia -it pixarninja/tf-gpu
#FROM tensorflow/tensorflow:nightly-gpu-py3-jupyter
FROM tensorflow/tensorflow:1.13.1-gpu-py3

# Install blocksparse and torch
RUN python -m pip install --upgrade pip
RUN pip install blocksparse
RUN pip install pillow
RUN pip install matplotlib

# Copy application and configure blocksparse
COPY ./app/blocksparse_setup.sh /scripts/blocksparse_setup.sh
RUN ./scripts/blocksparse_setup.sh

# Install OpenCV
RUN apt-get update
RUN apt-get install -y build-essential apt-utils

RUN apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev \
  libavformat-dev libswscale-dev
RUN  apt-get update && apt-get install -y python-dev python-numpy \
  python3 python3-pip python3-dev libtbb2 libtbb-dev \
  libjpeg-dev libjasper-dev libdc1394-22-dev \
  python-opencv libopencv-dev libav-tools python-pycurl \
  libatlas-base-dev gfortran webp qt5-default libvtk6-dev zlib1g-dev

RUN pip3 install numpy

RUN apt-get install -y python-pip
RUN pip install --upgrade pip

RUN cd ~/ &&\
    git clone https://github.com/Itseez/opencv.git &&\
    git clone https://github.com/Itseez/opencv_contrib.git &&\
    cd opencv && mkdir build && cd build && cmake  -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON -DWITH_XINE=ON -DBUILD_EXAMPLES=ON .. && \
    make -j4 && make install && ldconfig

# Set the appropriate link
RUN ln /dev/null /dev/raw1394

RUN pip install torchvision
