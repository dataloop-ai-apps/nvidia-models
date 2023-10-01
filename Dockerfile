FROM nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf1.15.5

COPY model/ /model/

RUN pip3 install --user dtlpy
RUN chmod -R 777 /model

ENV TLT_KEY=tlt_encode
