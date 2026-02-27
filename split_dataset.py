import os
import random
import shutil

DATASET_PATH = "data/PlantVillage"
OUTPUT_PATH = "data"
SPLIT_RATIO = 0.8
SEED = 123

random.seed(SEED)

train_path = os.path.join(OUTPUT_PATH, "train")
val_path = os.path.join(OUTPUT_PATH, "val")

os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

for class_name in os.listdir(DATASET_PATH):

    class_dir = os.path.join(DATASET_PATH, class_name)
    if not os.path.isdir(class_dir):
        continue

    images = os.listdir(class_dir)
    random.shuffle(images)

    split_index = int(len(images) * SPLIT_RATIO)

    train_images = images[:split_index]
    val_images = images[split_index:]

    for img in train_images:
        src = os.path.join(class_dir, img)
        dst_dir = os.path.join(train_path, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, dst_dir)

    for img in val_images:
        src = os.path.join(class_dir, img)
        dst_dir = os.path.join(val_path, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, dst_dir)

    print(f"{class_name}: {len(train_images)} train | {len(val_images)} val")

print("\nDataset split completed successfully.")