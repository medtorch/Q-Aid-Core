#FROM python:3.8-slim
FROM pytorch/pytorch

RUN mkdir /service
WORKDIR /service

COPY vqa .
COPY vqa/requirements.txt .

RUN apt-get update
RUN apt-get install python-opencv -y

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "main:app"]
