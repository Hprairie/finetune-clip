from clip_benchmark.datasets.builder import build_dataset
import pandas as pd
import os
from torchvision.datasets import CocoCaptions

# This is just for me 
dataset_path = os.environ['DATASETS']
annotations = "captions_val2017"
image_folder = "val2017"

image_path = f"{dataset_path}/coco/images" # set this to smth meaningful
ann_path = f"{dataset_path}/coco/annotations/{annotations}.json"
ds = CocoCaptions(
        root=image_path, 
        annFile=ann_path,
        transform=None,
)
coco = ds.coco
imgs = coco.loadImgs(coco.getImgIds())
future_df = {"filepath":[], "title":[]}
for img in imgs:
    caps = coco.imgToAnns[img["id"]]
    for cap in caps:
        future_df["filepath"].append(
            os.path.join(f'{dataset_path}/coco/images/{image_folder}', img["file_name"])
        )
        future_df["title"].append(cap["caption"])
pd.DataFrame.from_dict(future_df).to_csv(
  os.path.join(f"{dataset_path}/coco/annotations", f"{annotations}.csv"), index=False, sep="\t"
)
