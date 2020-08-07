import sys

import inference
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append("./MICCAI19-MedVQA")


class Question(BaseModel):
    image_b64: str
    question: str


app = FastAPI()


VQA = inference.VQA()


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
