FROM nvidia/cuda:10.1-base-ubuntu18.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl ca-certificates sudo git bzip2 libx11-6 \
    gcc g++ make cmake zlib1g-dev swig libsm6 libxext6 \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    wget llvm libncurses5-dev xz-utils tk-dev libxrender1\
    libxml2-dev libxmlsec1-dev libffi-dev libcairo2-dev libjpeg-dev libgif-dev

RUN adduser --disabled-password --gecos '' --shell /bin/bash user && chown -R user:user /home/user
RUN echo 'user ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers.d/90-pyrado
USER user
WORKDIR /home/user

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash Miniconda3-latest-Linux-x86_64.sh -b \
 && rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH /home/user/miniconda3/bin:$PATH

RUN conda update conda \
 && conda update --all

COPY --chown=user:user . SimuRLacra

WORKDIR /home/user/SimuRLacra
RUN conda env create -f Pyrado/environment.yml
RUN conda init bash
SHELL ["conda", "run", "-n", "pyrado", "/bin/bash", "-c"]

RUN echo "export PATH=/home/user/miniconda3/bin:$PATH" >> ~/.bashrc
RUN echo "conda activate pyrado" >> ~/.bashrc

RUN python setup_deps.py dep_libraries -j4
#RUN python setup_deps.py all --use-cuda -j4

RUN python setup_deps.py separate_pytorch -j4
RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH
ENV PYTHONPATH /home/user/SimuRLacra/RcsPySim/build/lib:/home/user/SimuRLacra/Pyrado/:$PYTHONPATH
RUN sudo apt-get install -y chromium-browser && sudo rm -rf /var/lib/apt/lists/*
