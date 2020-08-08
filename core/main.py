import sys

from fastapi import FastAPI
from pydantic import BaseModel

import vqa.inference


class Question(BaseModel):
    image_b64: str
    question: str


app = FastAPI()


VQA = vqa.inference.VQA()


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
