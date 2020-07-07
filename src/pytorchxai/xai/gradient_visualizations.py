from pytorchxai.xai.gradient_guided_backprop import GuidedBackprop


class GradientVisualization:
    def __init__(self, model):
        self.model = model

        self.guided_bp = GuidedBackprop(model)

    def generate(self, img, target):
        return self.guided_bp.generate(img, target)
