import numpy as np
from PIL import Image

from pytorchxai.xai.utils import preprocess_image


def create_image(width=244, height=244):
    width = int(width)
    height = int(height)

    rgb_array = np.random.rand(height, width, 3) * 255
    image = Image.fromarray(rgb_array.astype("uint8")).convert("RGB")
    prep = preprocess_image(image)

    target = 42

    return image, prep, target
