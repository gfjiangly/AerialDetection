ARG PYTORCH="1.1.0"
ARG CUDA="10.0"
ARG CUDNN="7.5"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6

# Install mmdetection
RUN conda install cython -y
RUN git clone https://github.com/dingjiansw101/AerialDetection.git /AerialDetection
RUN cd /AerialDetection && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e . 

# ARG PYTORCH="1.1.0"
# ARG CUDA="10.0"
# ARG CUDNN="7.5"

# FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
# ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
# ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Install r3det-on-mmdetection
# RUN conda clean --all
# RUN git clone https://github.com/SJTU-Thinklab-Det/r3det-on-mmdetection.git /r3det-on-mmdetection
# WORKDIR /r3det-on-mmdetection
# ENV FORCE_CUDA="1"
# RUN pip install cython --no-cache-dir
# RUN pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
# RUN pip install --no-cache-dir -e .
