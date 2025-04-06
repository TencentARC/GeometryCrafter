import torch
import torch.nn as nn
import sys

sys.path.append('third_party/moge')
from .moge.moge.model.moge_model import MoGeModel

class MoGe(nn.Module):
    
    def __init__(self, cache_dir):
        super().__init__()
        self.model = MoGeModel.from_pretrained(
            'Ruicheng/moge-vitl', cache_dir=cache_dir).eval()


    @torch.no_grad()
    def forward_image(self, image: torch.Tensor, **kwargs):
        # image: b, 3, h, w 0,1
        output = self.model.infer(image, resolution_level=9, apply_mask=False, **kwargs)
        points = output['points'].cpu() # b,h,w,3
        masks = output['mask'].cpu() # b,h,w
        return points, masks
