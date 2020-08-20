<p align="center">
  <img align="center" src="https://github.com/medtorch/Q-Aid/blob/master/misc/q_aid_logo_small1.png" alt="Q&Aid" width="75%">
</p>

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Q&Aid](https://github.com/medtorch/Q-Aid/workflows/Q&Aid/badge.svg)

# Core logic

## Introduction

The code contains scripts for testing and deploying the core logic behind Q&Aid.




## Prerequisites

```
pip install -r requirements.txt
```

## Run the server

```
uvicorn main:app --host 0.0.0.0 --port 8000
```

GET http://127.0.0.1:8000/models should return the list of available models.

## Models

### VQA

### Medical segmentation

### Medical labelling

### Chatbot using PyText
