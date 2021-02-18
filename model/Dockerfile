# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8

EXPOSE 8050

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Install pip requirements
ADD requirements.txt .
RUN python -m pip install -r requirements.txt
RUN pip install jupyter
ADD /src /src

#VOLUME ["/src/files"]
#VOLUME /src/files
#ADD /src/files /src/files

# jupyter notebook
# RUN pip install jupyter

WORKDIR /src
# WORKDIR /src/files
# RUN ["bash"]
# CMD ["python3","index.py"]

# CMD ["python3", "main.py"]
# During debugging, this entry point will be overridden. For more information, refer to https://aka.ms/vscode-docker-python-debug
