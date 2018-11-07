ARG BASEIMAGE=floydhub/pytorch:0.3.1-gpu.cuda9cudnn7-py3.35
FROM ${BASEIMAGE}

RUN apt-get update
RUN apt-get install -y libopencv-dev
RUN pip3 install --upgrade pip

RUN gpg --keyserver ha.pool.sks-keyservers.net --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4
RUN curl -o /usr/local/bin/gosu -SL "https://github.com/tianon/gosu/releases/download/1.4/gosu-$(dpkg --print-architecture)" \
    && curl -o /usr/local/bin/gosu.asc -SL "https://github.com/tianon/gosu/releases/download/1.4/gosu-$(dpkg --print-architecture).asc" \
    && gpg --verify /usr/local/bin/gosu.asc \
    && rm /usr/local/bin/gosu.asc \
    && chmod +x /usr/local/bin/gosu

ENV INSTALL_ROOT /opt

COPY ./entrypoint /entrypoint
RUN sed -i 's/\r//' /entrypoint
RUN chmod +x /entrypoint


#INSTALL RFCN-Pytorch
ENV RFCN_ROOT ${INSTALL_ROOT}/R-FCN.pytorch

RUN git clone --branch kaggle_submission https://github.com/arvindmvepa/R-FCN.pytorch ${RFCN_ROOT}
RUN cd ${RFCN_ROOT} && git pull && git checkout 875918fce6e702aefdd1e63fdb729b41288efbef
ENV RFCN_LIB ${RFCN_ROOT}/lib
COPY resnet152_rcnn.pth /opt/R-FCN.pytorch/data/pretrained_model/

WORKDIR ${RFCN_ROOT}
RUN pip3 install -r requirements.txt

WORKDIR ${RFCN_LIB}
RUN bash make.sh

RUN mkdir -p /work /home/user /notebooks
WORKDIR /notebooks

COPY ./run_jupyter.sh /run_jupyter.sh
RUN sed -i 's/\r//' /run_jupyter.sh
RUN chmod +x /run_jupyter.sh
COPY *.ipynb /notebooks/

ENV PYTHONPATH ${RFCN_LIB}:${RFCN_ROOT}:$PYTHONPATH

ENTRYPOINT ["/entrypoint"]
