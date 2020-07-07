from copy import deepcopy

import torch

class SaliencyMap:
    @staticmethod
    def prepare_model(model):
        model = deepcopy(model)

        for param in model.parameters():
            param.requires_grad = False

        # we would run the model in evaluation mode
        model.eval()

        return model

    @staticmethod
    def generate(img, model):
        model = SaliencyMap.prepare_model(model)


        # we need to find the gradient with respect to the input image,
        # so we need to call requires_grad_ on it
        img.requires_grad_()

        scores = model(img)

        # Get the index corresponding to the maximum score and the maximum score itself.
        score_max_index = scores.argmax()
        score_max = scores[0, score_max_index]

        score_max.backward()

        # Saliency would be the gradient with respect to the input image now.
        # But note that the input image has 3 channels, R, G and B.
        # To derive a single class saliency value for each pixel (i, j),
        # we take the maximum magnitude across all colour channels.

        saliency, _ = torch.max(img.grad.data.abs(), dim=1)

        return saliency
