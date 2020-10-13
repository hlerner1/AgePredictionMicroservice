FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

WORKDIR /app

COPY ./ /app

RUN pip install --upgrade pip

RUN pip --no-cache-dir install -r requirements.txt

RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y
