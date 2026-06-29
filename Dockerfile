FROM nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf1.15.5

# Install Python 3.10 and update alternatives
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install unzip
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

ENV PATH="/tmp/ngccli/ngc-cli/:$PATH"

ENV HOME="/tmp" \
    VS_CODE_VERSION="4.16.1"

# Install dtlpy with Python 3.10
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --user dtlpy

RUN python3 -m pip install seaborn==0.13.2

WORKDIR $HOME
RUN curl -fOL "https://github.com/coder/code-server/releases/download/v"$VS_CODE_VERSION"/code-server_"$VS_CODE_VERSION"_amd64.deb" && \
    dpkg -i "code-server_"$VS_CODE_VERSION"_amd64.deb" && \
    rm "code-server_"$VS_CODE_VERSION"_amd64.deb" && \
    code-server --install-extension ms-python.python && \
    chmod -R 777 /tmp

# Build command (replace YOUR_USERNAME with your Docker Hub username):
# docker build --no-cache -t YOUR_USERNAME/nvidia-tao-updated:1.0 -f Dockerfile .
# 
# Push command:
# docker push YOUR_USERNAME/nvidia-tao-updated:1.0
