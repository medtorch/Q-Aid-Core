import base64
import os
import requests
import json
from PIL import Image
from io import BytesIO
from skimage.io import imsave
import cv2
import numpy as np

samples = "./samples/"

requests_session = requests.Session()
server = "http://127.0.0.1:8000/segmentation"


for subdir, dirs, files in os.walk(samples):
    for f in files:
        path = subdir + f
        print(path)

        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()

        payload = {
            "image_b64": encoded_string,
        }

        r = requests_session.post(server, json=payload, timeout=10)

        data = json.loads(r.text)
        output = data["answer"]["hip"]

        for source in output:
            string = output[source]["segmentation"]
            decoded = base64.b64decode(string)

            np_img = np.frombuffer(decoded, dtype=np.uint8)
            np_img = np_img.reshape((256, 256, 3))

            image = Image.fromarray(np_img.astype('uint8'), 'RGB')

            image.save(os.path.join(subdir, f"result_{f}"), "PNG")
