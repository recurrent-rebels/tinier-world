FROM python:3.9

WORKDIR /src

## only run pip install if requirements.txt has changed
COPY requirements.txt .
RUN pip3 install -r requirements.txt


COPY . .


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7777"]
