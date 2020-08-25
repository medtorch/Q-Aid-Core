#!/bin/sh


pip install -r requirements.txt

git submodule init && git submodule update

cd models/model_vqa &&  git submodule init && git submodule update
cd -


