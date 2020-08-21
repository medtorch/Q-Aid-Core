from model_vqa.inference import VQA
from model_brain_segmentation.inference import Segmentation


class HealthIntelProviderLocal:
    def __init__(self, name, capabilities):
        self.name = name
        self.capabilities = capabilities

        self.cache = {}
        self.models = {}

        for feat in capabilities:
            self.cache[feat] = {}
            if feat == "vqa":
                self.models[feat] = VQA()
            elif feat == "segmentation":
                self.models[feat] = Segmentation()
            else:
                raise "not implemented"

    def vqa(self, question: str, image_b64: str, topic: str):
        if not self.supports("vqa", topic):
            raise NotImplementedError()

        results = {}
        try:
            result = self.models["vqa"].ask(question, image_b64)
            results["vqa"] = result
        except BaseException as e:
            print("vqa failed ", e)

        return results

    def segment(self, image_b64: str, topic: str):
        if not self.supports("segmentation", topic):
            raise NotImplementedError()

        results = {}
        try:
            result = self.models["segmentation"].ask(image_b64)
            results["segmentation"] = result
        except BaseException as e:
            print("vqa failed ", e)

        return results

    def supports(self, model: str, topic: str):
        if model not in self.capabilities:
            return False
        if topic not in self.capabilities[model]:
            return False

        return True
