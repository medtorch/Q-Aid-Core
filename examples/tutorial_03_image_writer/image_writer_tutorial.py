from pytorchxai.plugin.writer import TorchXAIWriter
from PIL import Image
from torchvision.transforms.functional import to_tensor
import torchvision

logdir = "./runs/image_data"
image = Image.open("/home/tudor/Downloads/5df126b679d7570ad2044f3e.jpeg")
size = 264

model = torchvision.models.vgg19(pretrained=True)
transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((size, size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                torchvision.transforms.Lambda(lambda x: x[None]),
            ]
        )

writer = TorchXAIWriter()
writer.add_saliency(image=image, model=model, transform=transform)