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
        results = {"aggregated": {}, "total": 0, "hip": {}}
        for provider in self.providers:
            if "vqa" not in provider.capabilities:
                continue
            if q.category not in provider.topics:
                continue
            results["hip"][provider.name] = provider.ask(q)
            results["total"] += 1

            for model in results["hip"][provider.name]:
                if model not in results["aggregated"]:
                    results["aggregated"][model] = {}
                val = results["hip"][provider.name][model].lower()
                if val not in results["aggregated"][model]:
                    results["aggregated"][model][val] = 0
                results["aggregated"][model][val] += 1
        return results

    def has_anomalies(self, q: ImageProto, prefilter):
        filter_q = "is there something abnormal in the image?"
        questions = {
            "what": "what is abnormal in the image?",
            "why": "why is this abnormal?",
            "where": "where is something abnormal?",
        }

        vqa = QuestionProto(
            image_b64=q.image_b64, question=filter_q, category=prefilter["category"]
        )

        has_anomaly = self.ask(vqa)

        results = {"has": 0, "total": has_anomaly["total"]}

        if has_anomaly["total"] == 0:
            return results

        if "yes" not in has_anomaly["aggregated"]["vqa"]:
            return results

        results["has"] = has_anomaly["aggregated"]["vqa"]["yes"]
        for qtype in questions:
            vqa = QuestionProto(
                image_b64=q.image_b64,
                question=questions[qtype],
                category=prefilter["category"],
            )
            res = self.ask(vqa)
            results[qtype] = res["aggregated"]["vqa"]

        return results


class Filter:
    def __init__(self):
        self.prefilter = Prefilter()
        self.router = ImageRouter()
        self.cache = {}

    def ask(self, q: ImageProto):
        h = hash_input(q.image_b64)
        if h in self.cache:
            return self.cache[h]

        valid = 0 == self.prefilter.ask(q.image_b64)

        result = {
            "valid": valid,
        }

        if valid:
            result["category"] = self.router.ask(q.image_b64)

        self.cache[h] = result

        return result
