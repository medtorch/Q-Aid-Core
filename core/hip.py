import model_vqa.inference

from proto import QuestionProto


class HealthIntelProviderLocal:
    def __init__(self, name, capabilities):
        self.name = name

        self.cache = {}
        self.models = {}

        for feat in capabilities:
            self.cache[feat] = {}
            if feat == "vqa":
                self.models[feat] = vqa.inference.VQA()
            else:
                raise "not implemented"

    def ask(q: QuestionProto):
        if "vqa" not in self.models:
            raise NotImplemented()

        results = {}
        try:
            result = self.models["vqa"].ask(q.question, q.image_b64)
            results["vqa"] = result
        except BaseException as e:
            print("vqa failed ", e)

        return results
