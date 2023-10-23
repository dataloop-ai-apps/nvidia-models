FROM nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf1.15.5

RUN pip3 install --user dtlpy
RUN #chmod -R 777 /model

ENV TLT_KEY=tlt_encode

ENV HOME="/tmp" \
    VS_CODE_VERSION="4.16.1"
WORKDIR $HOME
RUN curl -fOL "https://github.com/coder/code-server/releases/download/v"$VS_CODE_VERSION"/code-server_"$VS_CODE_VERSION"_amd64.deb" && \
    dpkg -i "code-server_"$VS_CODE_VERSION"_amd64.deb" && \
    rm "code-server_"$VS_CODE_VERSION"_amd64.deb" && \
    code-server --install-extension ms-python.python && \
    chmod -R 777 /tmp


# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/gpu/nvidia-tao:0.1.2 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/nvidia-tao:0.1.2
