FROM nvcr.io/nvidia/pytorch:21.06-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
    	ffmpeg \
        git \
        vim \
        wget \
		sudo \
		curl \
		ca-certificates \
		openssh-server \
		less \
		lv && rm -rf /var/lib/apt/lists/*

# Install pytorch
RUN pip install numpy pandas simplejson pyyaml coloredlogs scipy librosa musdb museval hydra-core==0.11.3
RUN pip install Cython asteroid norbert pytorch_lightning
RUN pip install jupyterlab


