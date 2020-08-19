import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
from torchvision import models
from torch import nn
import time
import base64


class ImageRouter:
    def __init__(self):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.label_mapping = {
            "XR_ELBOW": 0,
            "XR_FOREARM": 1,
            "XR_HAND": 2,
            "XR_HUMERUS": 3,
            "XR_SHOULDER": 4,
            "XR_WRIST": 5,
            "brain": 6,
            "breast": 7,
            "chest_xray": 8,
            "eyes": 9,
            "heart": 10,
        }

        self.reversed_label_mapping = {v: k for k, v in self.label_mapping.items()}

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
        self.model.classifier = nn.Linear(1024, 11)
        self.softmax = nn.Softmax()

        state_dict = torch.load("model_medical_label/saved_state_dict.pt", device)
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
        output = self.softmax(self.model(transformed_image))
        return self.reversed_label_mapping[torch.argmax(output).item()]
