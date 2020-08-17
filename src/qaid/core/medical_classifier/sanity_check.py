import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO
from collections import OrderedDict



input_size = 224

transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])



model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = model.classifier = nn.Sequential(
    OrderedDict([
        ("drop", nn.Dropout(0.2)),
        ("class", nn.Linear(num_ftrs, 2))
    ]))
my_softmax = nn.Softmax()

state_dict = torch.load("saved_state_dict.pt")
model.load_state_dict(state_dict)
model.eval()

print("Medical")
for medical_img in medical_imgs:
    response = requests.get(medical_img)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    transformed_image = transform(img).unsqueeze(0)
    output = my_softmax(model(transformed_image))
    print(output)

print("Nonmedical")
for non_medical_img in nonmedical_imgs:
    response = requests.get(non_medical_img)
    img = Image.open(BytesIO(response.content))
    transformed_image = transform(img).unsqueeze(0)
    output = my_softmax(model(transformed_image))
    print(output)
