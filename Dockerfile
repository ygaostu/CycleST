FROM tensorflow/tensorflow:2.6.1-gpu
RUN pip install --upgrade pip && pip install Pillow scipy