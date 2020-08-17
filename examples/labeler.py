from torchvision import transforms
from qaid.src.qaid.core.label_router import load_medical_router
import requests
from PIL import Image
from io import BytesIO
import torch

model = load_medical_router(path="../trained_weights/labeler_weights.pt")
model.eval()

medical_imgs = [
    "https://media.sciencephoto.com/image/c0371577/800wm/C0371577-Stroke,_MRI_brain_scan.jpg",
    "https://prod-images-static.radiopaedia.org/images/34839897/e0bfac31c00d077d18aca7ab33b495_gallery.jpeg",
    "https://prod-images-static.radiopaedia.org/images/157210/332aa0c67cb2e035e372c7cb3ceca2_jumbo.jpg",
    "https://image.freepik.com/photos-gratuite/technologie-rayon-chirurgie-x-ray-xray_1172-444.jpg",
    "https://media.wired.com/photos/5ba015a0ab6e142d95f93dac/125:94/w_1196,"
    "h_900,c_limit/R.Kim-eyescan-w.jpg",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ4IevMy0w_3XO3Wc"
    "-PNRB5lBgwqvoSSttiAw&usqp=CAU",
    "https://www.startradiology.com/uploads/images/english-class-x-elbow-fig-5-normal-anatomy-elbow-lateral-blanco.jpg"
    ]

label_mapping = {'XR_ELBOW': 0,
 'XR_FOREARM': 1,
 'XR_HAND': 2,
 'XR_HUMERUS': 3,
 'XR_SHOULDER': 4,
 'XR_WRIST': 5,
 'brain': 6,
 'breast': 7,
 'chest_xray': 8,
 'eyes': 9,
 'heart': 10,
}

reversed_label_mapping = {
    v: k for k,v in label_mapping.items()
}

input_size = 224

transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

my_softmax = torch.nn.Softmax()

for medical_img in medical_imgs:
    response = requests.get(medical_img)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    transformed_image = transform(img).unsqueeze(0)
    out = model(transformed_image)
    output = my_softmax(out)
    print(reversed_label_mapping[torch.argmax(output).item()])
