import argparse

import torchvision.transforms as T
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from pytorchxai.xai.utils import preprocess_image
from pytorchxai.xai.visualizations import GradientVisualization

pretrained_model = models.alexnet(pretrained=True)
writer = SummaryWriter()

parser = argparse.ArgumentParser()
parser.add_argument("--image")
parser.add_argument("--target_class")
args = parser.parse_args()

original_image = Image.open(args.image).convert("RGB")
writer.add_image("input", T.ToTensor()(original_image), 0)

images = []

original_image = original_image.resize((224, 224), Image.ANTIALIAS)
prep_img = preprocess_image(original_image)

print(prep_img.shape)
vis = GradientVisualization(pretrained_model)

output = vis.generate(original_image, prep_img, int(args.target_class))

for g in output:
    print("adding ", g)
    writer.add_image(g, output[g], 0)

writer.close()
