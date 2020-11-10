FROM python:3.6.12-slim-buster

RUN apt-get update
RUN apt-get -y install gcc ffmpeg libsm6 libxext6
RUN pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install pyyaml==5.1 pycocotools>=2.0.1 uvicorn fastapi opencv-python python-multipart pytest
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.6/index.html

WORKDIR /app
COPY . /app

EXPOSE 8888

CMD ["uvicorn", "app.app:app", "--reload", "--host", "0.0.0.0", "--port", "8888"]