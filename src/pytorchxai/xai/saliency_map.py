import torch
import torchvision
import torchvision.transforms as T


class SaliencyMap:
    model = torchvision.models.vgg19(pretrained=True)

    def __init__(self, writer):
        for param in self.model.parameters():
            param.requires_grad = False
        self.writer = writer

    def preprocess(self, image, size=224):
        transform = T.Compose(
            [
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.Lambda(lambda x: x[None]),
            ]
        )
        return transform(image)

    def generate(self, img):
        # preprocess the image
        self.writer.add_image("input", T.ToTensor()(img), 0)
        X = self.preprocess(img)

        # we would run the model in evaluation mode
        self.model.eval()

        # we need to find the gradient with respect to the input image,
        # so we need to call requires_grad_ on it
        X.requires_grad_()

        scores = self.model(X)
        self.writer.add_graph(self.model, X)

        # Get the index corresponding to the maximum score and the maximum score itself.
        score_max_index = scores.argmax()
        score_max = scores[0, score_max_index]

        score_max.backward()

        # Saliency would be the gradient with respect to the input image now.
        # But note that the input image has 3 channels, R, G and B.
        # To derive a single class saliency value for each pixel (i, j),
        # we take the maximum magnitude across all colour channels.

        saliency, _ = torch.max(X.grad.data.abs(), dim=1)
        self.writer.add_image("saliency", saliency, 0)

        return saliency
