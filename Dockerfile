FROM nvidia/cuda:11.3.0-devel-ubuntu20.04

ENV LANG en_US.utf8
ENV LC_ALL en_US.utf8

WORKDIR /root

RUN apt update
RUN apt install -y python3 python3-pip libsndfile1

COPY [ "./requirements.txt", "./requirements.txt" ]

RUN pip3 install --no-cache-dir -r ./requirements.txt

RUN pip3 install --no-cache-dir --upgrade google-cloud google-cloud-storage

COPY [ "./credentials.json", "./credentials.json" ]

ENV GOOGLE_APPLICATION_CREDENTIALS ./credentials.json
ENV GOOGLE_CLOUD_PROJECT diffusion-project

COPY [ "./src", "./src" ]


ENTRYPOINT [ "python3", "./src/ddpm.py" ]