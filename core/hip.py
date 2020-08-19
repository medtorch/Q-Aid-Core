from model_vqa.inference import VQA

from proto import QuestionProto


class HealthIntelProviderLocal:
    def __init__(self, name, capabilities, topics):
        self.name = name
        self.capabilities = capabilities
        self.topics = topics

        self.cache = {}
        self.models = {}

        for feat in capabilities:
            self.cache[feat] = {}
            if feat == "vqa":
                self.models[feat] = VQA()
            else:
                raise "not implemented"

    def ask(self, q: QuestionProto):
        if "vqa" not in self.models:
            raise NotImplementedError()
        if q.category not in self.topics:
            raise NotImplementedError()

        results = {}
        try:
            result = self.models["vqa"].ask(q.question, q.image_b64)
            results["vqa"] = result
        except BaseException as e:
            print("vqa failed ", e)

        return results
