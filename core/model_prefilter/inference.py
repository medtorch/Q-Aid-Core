import torch
from torchvision import transforms, models
from PIL import Image
from io import BytesIO
import base64
import time
from collections import OrderedDict


class Prefilter:
    def __init__(self):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        input_size = 224

        self.transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.model = models.densenet121(pretrained=False)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = torch.nn.Sequential(
            OrderedDict(
                [
                    ("drop", torch.nn.Dropout(0.2)),
                    ("class", torch.nn.Linear(num_ftrs, 2)),
                ]
            )
        )

        state_dict = torch.load("model_prefilter/medical_binary_classifier.pt", device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        x = torch.randn((1, 3, 224, 224))
        for i in range(3):
            start = time.time()
            self.model(x)
            print("Time elapsed: {}".format(time.time() - start))

    def ask(self, image_b64):
        decoded = base64.b64decode(image_b64)
        img = Image.open(BytesIO(decoded)).convert("RGB")

        transformed_image = self.transform(img).unsqueeze(0)
        return torch.argmax(torch.sigmoid(self.model(transformed_image))).item()
