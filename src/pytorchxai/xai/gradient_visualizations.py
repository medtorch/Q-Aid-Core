from pytorchxai.xai.gradient_guided_backprop import GuidedBackprop
from pytorchxai.xai.gradient_scorecam import ScoreCam
from pytorchxai.xai.gradient_vanilla_backprop import VanillaBackprop


class GradientVisualization:
    def __init__(self, model):
        self.model = model

        self.visualizations = [
            GuidedBackprop(model),
            VanillaBackprop(model),
            ScoreCam(model),
        ]

    def generate(self, img, target):
        results = {}
        for v in self.visualizations:
            results.update(v.generate(img, target))
        return results
