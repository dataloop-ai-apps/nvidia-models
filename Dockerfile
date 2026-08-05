FROM nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf1.15.5

RUN apt install unzip
# RUN wget "https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip" -P /tmp/ngccli
# RUN unzip -u /tmp/ngccli/ngccli_cat_linux.zip -d /tmp/ngccli/

ENV PATH="/usr/local/nvidia/bin:/tmp/ngccli/ngc-cli/:$PATH"

ENV HOME="/tmp" \
    VS_CODE_VERSION="4.16.1"
#--trusted-host pypi.org used for local build
RUN pip3 install --user --trusted-host pypi.org --trusted-host files.pythonhosted.org dtlpy==1.122.13 "urllib3<2"
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org seaborn==0.13.2

WORKDIR $HOME
RUN curl -fOLk "https://github.com/coder/code-server/releases/download/v"$VS_CODE_VERSION"/code-server_"$VS_CODE_VERSION"_amd64.deb" && \
    dpkg -i "code-server_"$VS_CODE_VERSION"_amd64.deb" && \
    rm "code-server_"$VS_CODE_VERSION"_amd64.deb" && \
    code-server --install-extension ms-python.python && \
    chmod -R 777 /tmp


# docker build --no-cache -t hub.dataloop.ai/customerhub/piper/agent/runner/gpu/nvidia-tao:0.1.4 -f Dockerfile .
# docker push hub.dataloop.ai/customerhub/piper/agent/runner/gpu/nvidia-tao:0.1.4
