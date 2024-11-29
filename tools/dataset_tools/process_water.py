from glob import glob
classes_mapping = {
    "CDUWD-1":1,
    "CDUWD-2":2,
    "CDUWD-3":3,
    "CDUWD-4":4,
    "CDUWD-5":5,
    "CDUWD-6":0,
}
all_images = glob("data/raw_water_dataset/512/CDUWD-6/images/*.png")

all_labels = [image_path.replace("images","labels")for image_path in all_images]

for image_path,label_path in zip(all_images,all_labels):
    pass