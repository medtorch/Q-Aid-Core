<p align="center">
  <img align="center" src="https://github.com/medtorch/Q-Aid/blob/master/misc/q_aid_logo_small1.png" alt="Q&Aid" width="75%">
</p>

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Q&Aid](https://github.com/medtorch/Q-Aid/workflows/Q&Aid/badge.svg)

## Features

- :fire: Collection of healthcare AI models under [core](core), created using PyTorch.
- :key: Served using [FastAPI](https://fastapi.tiangolo.com/).
- :cyclone: Full deployment scripts for AWS.
- :zap: Compatible React-Native app under [app](app) folder.

## Installation

```
pip install -r requirements.txt
```

## Usage

### Models

Read more about the models [here](https://github.com/medtorch/Q-Aid-Models).
### App

Read more about the app [here](https://github.com/medtorch/Q-Aid-App).

### Server

Run the server using `Docker`
```
docker build -t qaid/model_api .
docker run -p 8000:8000 qaid/model_api
```

Or using `uvicorn`
```
cd core
uvicorn main:app
```
### AWS deployment
Seet the [AWS README](aws_backend/README.md).


## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md).

## License
[MIT License](https://choosealicense.com/licenses/mit/)
