import sys

from fastapi import FastAPI
from pydantic import BaseModel

import vqa.inference
import medical_classifier.inference
import label_router.inference

import hashlib


def hash_input(val):
    hash_object = hashlib.sha1(val.encode("utf-8"))
    return hash_object.digest()


class Question(BaseModel):
    image_b64: str
    question: str


class Image(BaseModel):
    image_b64: str


app = FastAPI()

VQA = vqa.inference.VQA()
Prefilter = medical_classifier.inference.Prefilter()
ImageRouter = label_router.inference.ImageRouter()

CACHE = {
    "vqa": {},
    "router": {},
    "prefilter": {},
}


@app.get("/models")
def get_models():
    return ["vqa", "router", "prefilter"]


@app.post("/vqa")
def vqa_query(q: Question):
    try:
        result = VQA.ask(q.question, q.image_b64)
        return {"answer": result}
    except BaseException as e:
        return {"error": str(e)}


@app.post("/prefilter")
def prefilter_query(q: Image):
    try:
        h = hash_input(q.image_b64)
        if h in CACHE["prefilter"]:
            return {"answer": CACHE["prefilter"][h]}

        result = Prefilter.ask(q.image_b64)
        CACHE["prefilter"][h] = result

        return {"answer": result}
    except BaseException as e:
        return {"error": str(e)}


@app.post("/router")
def router_query(q: Image):
    try:
        h = hash_input(q.image_b64)
        if h in CACHE["router"]:
            return {"answer": CACHE["router"][h]}

        result = ImageRouter.ask(q.image_b64)
        CACHE["router"][h] = result

        return {"answer": result}
    except BaseException as e:
        print("err ", e)
        return {"error": str(e)}
