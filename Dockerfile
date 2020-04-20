FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
# install anaconda 5.2.0
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]

# install pytorch 1.1 and cudatoolkit
RUN conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

# clone and install openvqa dependencies
RUN mkdir /workspace && \
    cd /workspace && \    
    git clone https://github.com/MILVLG/openvqa.git && \
    cd openvqa &&\
    pip install -r requirements.txt && \
    wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz && \
    pip install en_vectors_web_lg-2.1.0.tar.gz && \
    rm en_vectors_web_lg-2.1.0.tar.gz && \
    cd /
    
# delete openvqa repo
RUN rm -r /workspace/openvqa

WORKDIR /workspace

