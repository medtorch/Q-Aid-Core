import numpy as np
import torch
from medpy.filter.binary import largest_connected_component
import base64
from model_brain_segmentation.unet import UNet
from model_brain_segmentation.utils import gray2rgb, outline, normalize_volume
from io import BytesIO

from PIL import Image


class Segmentation:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        weights = "model_brain_segmentation//weights/unet.pt"

        self.model = UNet(in_channels=3, out_channels=1)
        state_dict = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

    def transform(self, img):
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img.astype(np.float32))
        img = img.unsqueeze(0)
        return img

    def ask(self, image_b64):
        decoded = base64.b64decode(image_b64)
        img = Image.open(BytesIO(decoded)).convert("RGB")
        img = normalize_volume(np.array(img))

        img = self.transform(img)
        img = img.to(self.device)
        pred = self.model(img)

        seg_mask = pred.squeeze(0).squeeze(0).detach().cpu().numpy()
        seg_mask = np.round(seg_mask).astype(int)

        if len(seg_mask[seg_mask != 0]) != 0:
            seg_mask = largest_connected_component(seg_mask)
        else:
            seg_mask = np.array(seg_mask, dtype=bool)

        initial_image = img.reshape((3, 256, 256))[1]
        initial_image = initial_image.reshape((256, 256)).detach().cpu().numpy()

        initial_image = gray2rgb(initial_image)
        outlined_img = outline(initial_image, seg_mask, color=[255, 0, 0])

        return outlined_img
