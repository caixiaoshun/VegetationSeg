import shutil
from glob import glob
import os
import numpy as np
from rich.progress import track
from PIL import Image
from sklearn.model_selection import train_test_split

def create_dataset(image_list,ann_list,image_dir,ann_dir):
    os.makedirs(image_dir,exist_ok=True)
    os.makedirs(ann_dir,exist_ok=True)
    for image_path,ann_path in zip(image_list,ann_list):
        base_name = os.path.basename(image_path)

        new_image_path = os.path.join(image_dir,base_name)
        new_ann_path = os.path.join(ann_dir,base_name)

        shutil.move(image_path,new_image_path)
        shutil.move(ann_path,new_ann_path)

classes_mapping = {
    "CDUWD-1": 1,
    "CDUWD-2": 2,
    "CDUWD-3": 3,
    "CDUWD-4": 4,
    "CDUWD-5": 5,
    "CDUWD-6": 0,
}
all_images = glob("data/raw_water_dataset/512/CDUWD-6/images/*.png")

all_labels = [image_path.replace("images", "labels") for image_path in all_images]

target_image_dir = "data/raw_water_dataset/all_dataset/images/"
target_label_dir = "data/raw_water_dataset/all_dataset/labels/"

os.makedirs(target_image_dir, exist_ok=True)
os.makedirs(target_label_dir, exist_ok=True)

for image_path, label_path in zip(all_images, all_labels):
    exists_images = glob("data/raw_water_dataset/all_dataset/images/*.png")

    base_name = os.path.basename(image_path)
    if image_path not in exists_images:
        mask = np.array(Image.open(label_path))

        assert list(np.unique(mask)) in [
            [0],
            [1],
            [0, 1],
            [1, 0],
        ], f"The mask image is not binary (it should only contain 0s and 1s),actually is {set(np.unique(mask))}"

        classes_str = image_path.split(os.path.sep)[-3]
        classes = classes_mapping[classes_str]
        mask = np.where(mask == 1, classes, mask)

        mask = Image.fromarray(mask)
        mask.save(os.path.join(target_label_dir, base_name))
        shutil.copy(image_path, os.path.join(target_image_dir, base_name))
    else:

        exists_label_path = os.path.join(target_label_dir, base_name)
        exists_mask = np.array(Image.open(exists_label_path))

        mask = np.array(Image.open(label_path))
        assert list(np.unique(mask)) in [
            [0],
            [1],
            [0, 1],
            [1, 0],
        ], f"The mask image is not binary (it should only contain 0s and 1s),actually is {set(np.unique(mask))}"
        classes_str = image_path.split(os.path.sep)[-3]
        classes = classes_mapping[classes_str]

        exists_mask = np.where(mask == 1, classes, exists_mask)

        exists_mask = Image.fromarray(exists_mask)
        exists_mask.save(exists_label_path)

print("process done.")

exists_images = glob("data/raw_water_dataset/all_dataset/images/*.png")
exists_labels = [image_path.replace("images", "labels") for image_path in exists_images]
X_train, X_test, y_train, y_test = train_test_split(exists_images,exists_labels,test_size=0.2,random_state=42,shuffle=True)


create_dataset(X_train,y_train,"data/water/img_dir/train","data/water/ann_dir/train")
create_dataset(X_test,y_test,"data/water/img_dir/val","data/water/ann_dir/val")