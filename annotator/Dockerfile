# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8

EXPOSE 8050

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Install pip requirements
ADD requirements.txt .
RUN python -m pip install -r requirements.txt

#jupyter notebook 
RUN pip install jupyter

COPY /app /app
# COPY /jupyter_notebooks /jupyter_notebooks
# COPY /csv /csv

WORKDIR /app
CMD ["python3","index.py"]
# During debugging, this entry point will be overridden. For more information, refer to https://aka.ms/vscode-docker-python-debug



