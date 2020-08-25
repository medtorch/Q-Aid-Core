<p align="center">
  <img align="center" src="https://github.com/medtorch/Q-Aid/blob/master/misc/q_aid_logo_small1.png" alt="Q&Aid" width="75%">
</p>


# Core logic

## Introduction

Scripts for testing and deploying the core logic behind Q&Aid.


## Prerequisites

```
./setup.sh
```

Download the VQA model from https://drive.google.com/file/d/1dqJjthrbdnIs41ZdC_ZGVQnoZbuGMNCR/view?usp=sharing
and save it to the path models/model_vqa/MICCAI19-MedVQA/saved_models/BAN_MEVF/model_epoch19.pth

## Run the server

```
uvicorn main:app --host 0.0.0.0 --port 8000
```

GET http://127.0.0.1:8000/models should return the list of available models.

## Tests

Run the scripts in the `tests` folder for checking each model.
The tests require a running server instance.

