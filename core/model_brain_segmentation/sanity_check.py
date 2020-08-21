import base64
import os

from inference import Segmentation

samples = "./samples/"
segm = Segmentation()


for subdir, dirs, files in os.walk(samples):
    for f in files:
        path = subdir + f
        print(path)
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()

        segm.ask(encoded_string)
