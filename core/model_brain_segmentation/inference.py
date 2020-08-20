import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from medpy.filter.binary import largest_connected_component
from skimage.io import imsave
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.io import imread

from unet import UNet
from utils import dsc, gray2rgb, outline, normalize_volume

def load_data(samples):
    for sample in samples:
        sample = sample.transpose(2, 0, 1)
        sample = torch.from_numpy(sample.astype(np.float32))
        sample = sample.unsqueeze(0)
        yield sample

def main(args):
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    path = args.path
    path_results = args.results

    samples_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    samples = [
        normalize_volume(
            np.array(
                imread(os.path.join(path, sample_name))
            )
        )
        for sample_name in samples_names
    ]

    with torch.set_grad_enabled(False):
        unet = UNet(in_channels=3, out_channels=1)
        state_dict = torch.load(args.weights, map_location=device)
        unet.load_state_dict(state_dict)
        unet.eval()
        unet.to(device)

        data = load_data(samples)
        for idx, sample in enumerate(data):
            sample = sample.to(device)
            pred = unet(sample)

            seg_mask = pred.squeeze(0).squeeze(0).detach().cpu().numpy()
            seg_mask = np.round(seg_mask).astype(int)

            if(len(seg_mask[seg_mask!=0]) !=0):
                seg_mask = largest_connected_component(seg_mask)
            else:
                seg_mask = np.array(seg_mask, dtype=bool)

            initial_image = sample.reshape((3, 256, 256))[1]
            initial_image = initial_image.reshape(
                (256, 256)
                ).detach().cpu().numpy()

            initial_image = gray2rgb(initial_image)
            mask = outline(initial_image, seg_mask, color=[255, 0, 0])

            outlined_img = outline(initial_image, seg_mask, color=[255, 0, 0])

            imsave(os.path.join(path_results, f'outlined_img_{idx}.png'), outlined_img)
            imsave(os.path.join(path_results, f'segmask_{idx}.png'), seg_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference for segmentation of brain MRI"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )

    parser.add_argument(
        "--weights", type=str, required=True, help="path to weights file"
    )
    parser.add_argument(
        "--path", type=str, default="./samples", help="path to samples"
    )

    parser.add_argument(
        "--results", type=str, default="./results", help="path to segmentation results"
    )

    args = parser.parse_args()
    main(args)
