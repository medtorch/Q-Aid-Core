from ..xai.cam_scorecam import ScoreCam
from .densenet import get_transforms

def apply_scorecam(model, img):
    cam = ScoreCam(model)
    transform = get_transforms()["val"]
    return cam.generate(orig_image=img, input_image=transform(img))
