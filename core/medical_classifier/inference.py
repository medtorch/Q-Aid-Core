import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import base64
import cv2

class Prefilter:
    def __init__(self):

        input_size = 224

        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.model = torch.jit.load("medical_classifier/scripted_model.pt")

    def ask(self, image_b64):
        decoded = base64.b64decode(image_b64)
        img = Image.open(BytesIO(decoded)).convert("RGB")

        transformed_image = self.transform(img).unsqueeze(0)
        print("pricess image ", transformed_image)
        return torch.sigmoid(self.model(transformed_image))
