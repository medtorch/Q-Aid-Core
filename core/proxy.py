from model_prefilter.inference import Prefilter
from model_medical_label.inference import ImageRouter

from helpers import hash_input
from proto import QuestionProto, ImageProto


class Proxy:
    def __init__(self):
        self.providers = []

    def register(self, handler):
        print("registering ", handler.name)
        self.providers.append(handler)

    def sources(self,):
        result = []
        for provider in self.providers:
            result.append(provider.name)
        return result

    def ask(self, q: QuestionProto):
        results = {}
        for provider in self.providers:
            if "vqa" not in provider.capabilities:
                continue
            results[provider.name] = provider.ask(q)
        return results


class Filter:
    def __init__(self):
        self.prefilter = Prefilter()
        self.router = ImageRouter()
        self.cache = {}

    def ask(self, q: ImageProto):
        h = hash_input(q.image_b64)
        if h in self.cache:
            return self.cache["prefilter"][h]

        results = {
            "valid": (0 == self.prefilter.ask(q.image_b64)),
            "cat": self.router.ask(q.image_b64),
        }
        self.cache[h] = results
        return results
