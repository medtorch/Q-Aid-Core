from model_prefilter.inference import Prefilter
from model_medical_label.inference import ImageRouter

from helpers import hash_input


class Filter:
    def __init__(self):
        self.prefilter = Prefilter()
        self.router = ImageRouter()
        self.cache = {}

    def ask(self, image_b64: str):
        h = hash_input(image_b64)
        if h in self.cache:
            return self.cache[h]

        valid = 0 == self.prefilter.ask(image_b64)

        result = {
            "valid": valid,
        }

        if valid:
            result["topic"] = self.router.ask(image_b64)

        self.cache[h] = result

        return result


class Proxy:
    def __init__(self):
        self.providers = []
        self.filter = Filter()

    def register(self, handler):
        print("registering ", handler.name)
        self.providers.append(handler)

    def sources(self):
        result = []
        for provider in self.providers:
            result.append(provider.name)
        return result

    def aggregate(self, results):
        data = results["hip"]
        results["total"] = 0
        results["aggregated"] = {}

        for provider in data:
            results["total"] += 1
            for model in data[provider]:
                if model not in results["aggregated"]:
                    results["aggregated"][model] = {}
                val = data[provider][model].lower()
                if val not in results["aggregated"][model]:
                    results["aggregated"][model][val] = 0
                results["aggregated"][model][val] += 1
        return results

    def prefilter(self, image_b64: str):
        return self.filter.ask(image_b64)

    def ask(self, question: str, image_b64: str, topic: str):
        results = {"hip": {}}
        for provider in self.providers:
            if not provider.supports("vqa", topic):
                continue

            results["hip"][provider.name] = provider.vqa(question, image_b64, topic)

        return results

    def segment(self, image_b64: str, topic: str):
        results = {"hip": {}}
        for provider in self.providers:
            if not provider.supports("segmentation", topic):
                continue
            results["hip"][provider.name] = provider.segment(image_b64, topic)

        return results

    def anomalies(self, image_b64: str, topic: str):
        filter_q = "is there something abnormal in the image?"
        questions = {
            "what": "what is abnormal in the image?",
            "why": "why is this abnormal?",
            "where": "where is something abnormal?",
        }

        has_anomaly = self.ask(filter_q, image_b64, topic)
        has_anomaly = self.aggregate(has_anomaly)

        results = {"has": 0, "total": has_anomaly["total"]}

        if has_anomaly["total"] == 0:
            return results

        if "yes" not in has_anomaly["aggregated"]["vqa"]:
            return results

        results["has"] = has_anomaly["aggregated"]["vqa"]["yes"]
        for qtype in questions:
            res = self.ask(questions[qtype], image_b64, topic)
            res = self.aggregate(res)

            results[qtype] = res["aggregated"]["vqa"]

        return results
