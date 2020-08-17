import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
import base64
import time


class Prefilter:
    def __init__(self):

        input_size = 224

        self.transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.model = torch.jit.load("medical_classifier/scripted_model.pt")
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
