FROM tensorflow/tensorflow:latest-gpu-py3-jupyter
RUN apt update
RUN apt install -y ffmpeg
RUN pip install --upgrade pip
RUN pip install pandas progressbar2 matplotlib