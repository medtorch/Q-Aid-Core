import argparse

from PIL import Image
from pytorchxai.xai.saliency_map import SaliencyMap
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--path")
args = parser.parse_args()

writer = SummaryWriter()
saliency_map = SaliencyMap(writer)

input_img = Image.open(args.path)
saliency_tensor = saliency_map.generate(input_img)

writer.close()
