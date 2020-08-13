import sys

from fastapi import FastAPI
from pydantic import BaseModel

import vqa.inference
import medical_classifier.inference


class Question(BaseModel):
    image_b64: str
    question: str


class Image(BaseModel):
    image_b64: str


app = FastAPI()

VQA = vqa.inference.VQA()
Prefilter = medical_classifier.inference.Prefilter()


@app.get("/models")
def get_models():
    return ["vqa"]


@app.post("/vqa")
def vqa_query(q: Question):
    try:
        result = VQA.ask(q.question, q.image_b64)
        return {"answer": result}
    except BaseException as e:
        return {"error": str(e)}


@app.post("/prefilter")
def vqa_query(q: Image):
    try:
        result = Prefilter.ask(q.image_b64)
        print("got result ", result)
        return {"answer": result}
    except BaseException as e:
        return {"error": str(e)}
