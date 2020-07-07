import torch.utils.tensorboard as tb
from pytorchxai.xai.saliency_map import SaliencyMap
from torchvision.transforms import ToTensor


class TorchXAIWriter(tb.SummaryWriter):
    def __init__(self, log_dir=None, comment="", purge_step=None, max_queue=10, flush_secs=120,
                 filename_suffix=""):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)

    def add_saliency(self, image, model, transform):
        self.add_image(tag="saliency_ref", img_tensor=ToTensor()(image))
        image = transform(image)
        saliency_img = SaliencyMap.generate(image, model)

        self.add_image(tag="saliency_img", img_tensor=saliency_img)
