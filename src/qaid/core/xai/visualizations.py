from .cam_gradcam import GradCam
from .cam_scorecam import ScoreCam
from .gradient_guided_backprop import GuidedBackprop
from .gradient_guided_gradcam import GuidedGradCam
from .gradient_integrated_grad import IntegratedGradients
from .gradient_smooth_grad import SmoothGrad
from .gradient_vanilla_backprop import VanillaBackprop


class GradientVisualization:
    def __init__(self, model):
        self.model = model

        self.visualizations = [
            GuidedBackprop(model),
            VanillaBackprop(model),
            ScoreCam(model),
            GradCam(model),
            GuidedGradCam(model),
            IntegratedGradients(model),
            SmoothGrad(model),
        ]

    def generate(self, orig_image, input_image, target):
        results = {}
        for v in self.visualizations:
            results.update(v.generate(orig_image, input_image, target))
        return results
