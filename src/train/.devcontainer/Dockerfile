# For more information, please refer to https://aka.ms/vscode-docker-python
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# 挂载项目，则不需要复制requirements.txt
# Install pip requirements
# COPY requirements.txt .
# # RUN python -m pip install -r requirements.txt
RUN apt update
RUN apt install git -y
RUN apt-get install libgl1 -y
RUN apt-get install glib2.0-dev -y
RUN apt install vim -y
RUN pip install --no-cache-dir --index-url https://pypi.nvidia.com --index-url https://pypi.org/simple pytorch-quantization==2.1.3
RUN pip install lightning==2.4.0
RUN pip install torchinfo==1.8.0
RUN pip install matplotlib==3.10.0
RUN pip install albumentations==1.4.24
RUN pip install torch-pruning==1.4.1
RUN pip install tensorboard==2.18.0
RUN pip install jupyter==1.1.1
RUN pip install opencv-python==4.10.0.84
RUN pip install onnx==1.17.0
RUN pip install onnx-simplifier==0.4.36
RUN pip install pandas==2.2.3

# WORKDIR /app
# COPY . /app

# # Creates a non-root user with an explicit UID and adds permission to access the /app folder
# # For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# CMD ["python", "main.py"]
