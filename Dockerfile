FROM tensorflow/tensorflow:latest-gpu-py3-jupyter
RUN apt-get update
RUN apt-get install -y ffmpeg
RUN pip install --upgrade pip
RUN pip install pandas progressbar2 matplotlib
