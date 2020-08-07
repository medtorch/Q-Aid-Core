import base64
import sys
from pathlib import Path

import base_model
import cv2
import dataset_RAD
import numpy as np
import torch
import torchvision.transforms.functional as TF
import utils
from bunch import Bunch

model_root = Path("MICCAI19-MedVQA")
sys.path.append("./MICCAI19-MedVQA")

torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VQA:
    def __init__(self):
        model_path = model_root / "saved_models/BAN_MEVF/model_epoch19.pth"

        data_RAD = model_root / "data_RAD"
        dictionary = dataset_RAD.Dictionary.load_from_file(data_RAD / "dictionary.pkl")

        args = Bunch(
            RAD_dir=str(data_RAD),
            autoencoder=True,
            maml=True,
            feat_dim=64,
            op="c",
            num_hid=1024,
            rnn="LSTM",
            gamma=2,
            ae_model_path="pretrained_ae.pth",
            maml_model_path="pretrained_maml.weights",
            activation="relu",
            dropout=0.5,
            eps_cnn=1e-5,
            momentum_cnn=0.05,
            map_location=device,
        )

        self.eval_dset = dataset_RAD.VQAFeatureDataset("test", args, dictionary)
        self.model = base_model.build_BAN(self.eval_dset, args)

        model_data = torch.load(model_path, device)
        self.model.train(False)
        self.model.load_state_dict(model_data.get("model_state", model_data))

    def load_image_tensors(self, base64_img):
        nparr = np.fromstring(base64.b64decode(base64_img), np.uint8)
        xray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        xray_ae = cv2.resize(xray, (128, 128))
        xray_maml = cv2.resize(xray, (84, 84))
        xray_ae_t = TF.to_tensor(np.array(xray_ae))
        xray_maml_t = TF.to_tensor(np.array(xray_maml))

        xray_maml_t = xray_maml_t.unsqueeze(1).to(device)
        xray_ae_t = xray_ae_t.unsqueeze(1).to(device)
        return [xray_maml_t, xray_ae_t]

    def tokenize(self, text, dataset, max_length=12):
        tokens = dataset.dictionary.tokenize(text, False)
        tokens = tokens[:max_length]
        if len(tokens) < max_length:
            # Note here we pad in front of the sentence
            padding = [dataset.dictionary.padding_idx] * (max_length - len(tokens))
            tokens = tokens + padding
        utils.assert_eq(len(tokens), max_length)
        tokens = torch.tensor([tokens]).to(device)
        return tokens

    def ask(self, question, image_b64):
        image_tensors = self.load_image_tensors(image_b64)
        tokens = self.tokenize(question, self.eval_dset)

        features, _ = self.model(image_tensors, tokens)
        logits = self.model.classifier(features)
        prediction = torch.max(logits, 1)[1].data

        result = self.eval_dset.label2ans[prediction.item()]

        return result
