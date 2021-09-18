FROM python:3.8-slim-buster

ENV PROJECT_DIR trainer
WORKDIR /${PROJECT_DIR}
ADD requirements.txt /${PROJECT_DIR}/

RUN apt update && \
    apt -y install build-essential && \
    apt-get -y update && \
    apt-get -y install manpages-dev libopencv-dev && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir pycocotools

CMD ["/bin/bash"]