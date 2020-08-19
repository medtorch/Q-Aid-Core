import sys

from fastapi import FastAPI

from helpers import hash_input
from proto import QuestionProto, ImageProto
from proxy import Proxy, Filter
from mocks import generate_mocks

app = FastAPI()
image_filter = Filter()

proxy = Proxy()

for hip_mock in generate_mocks():
    proxy.register(hip_mock)


@app.get("/sources")
def get_sources():
    return proxy.sources()


@app.post("/question")
def vqa_query(q: QuestionProto):
    try:
        result = proxy.ask(q.question, q.image_b64)
        return {"answer": result}
    except BaseException as e:
        return {"error": str(e)}


@app.post("/prefilter")
def prefilter_query(q: ImageProto):
    try:
        result = image_filter.ask(q)
        return {"answer": result}
    except BaseException as e:
        return {"error": str(e)}
