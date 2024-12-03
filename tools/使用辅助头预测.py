from mmseg.apis import init_model,inference_model
from PIL import Image
from vegseg.datasets import GrassDataset
import numpy as np
from typing import Tuple, List
from vegseg.models import DistillEncoderDecoder

def get_palette() -> List[int]:
    """
    get palette of dataset.
    return:
        palette: list of palette.
    """
    palette = []
    palette_list = GrassDataset.METAINFO["palette"]
    for palette_item in palette_list:
        palette.extend(palette_item)
    return palette

config = "work_dirs/experiment_p/experiment_p.py"
checkpoint = "work_dirs/experiment_p/best_mIoU_iter_22770.pth"
device = "cuda:4"
image_path = "data/grass/img_dir/train/8.tif"
model: DistillEncoderDecoder = init_model(config=config, checkpoint=checkpoint, device=device)
model.decode_head = model.auxiliary_head
result = inference_model(model, image_path)
pred = result.pred_sem_seg.data.cpu().numpy()[0].astype(np.uint8)
print(pred.shape,)
auxiliary_img = Image.fromarray(pred).convert('P')
palette = get_palette()
auxiliary_img.putpalette(palette)
auxiliary_img.save("auxiliary.png")

