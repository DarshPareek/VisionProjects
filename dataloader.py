import os
import numpy as np
import pandas as pd
from PIL import Image
import pickle


class ImageObject:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.red = data[:, :, 0]
        self.green = data[:, :, 1]
        self.blue = data[:, :, 2]
        self.grayscale = np.dot(data[..., :3], [0.299, 0.587, 0.114])

    def __repr__(self):
        return (
            f"<ImageObject label={self.label} grayscale_shape={self.grayscale.shape}>"
        )


TRAIN_FOLDER = "./train"
TEST_FOLDER = "./test"
CSV_PATH = "./trainLabels.csv"
IMG_SIZE = (32, 32)
train_data = []
test_data = []
print("Loading Training Data...")
try:
    df = pd.read_csv(CSV_PATH)
    label_map = pd.Series(df.label.values, index=df.id).to_dict()
    for filename in os.listdir(TRAIN_FOLDER):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            filepath = os.path.join(TRAIN_FOLDER, filename)
            try:
                img = Image.open(filepath).convert("RGB")
                img = img.resize(IMG_SIZE)
                img_array = np.array(img)
                d_keys = filename[:-4]
                label = label_map.get(int(d_keys), "Unknown")
                obj = ImageObject(img_array, label)
                train_data.append(obj)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
except FileNotFoundError:
    print(f"CRITICAL: Could not find CSV file at {CSV_PATH}")

print("Loading Test Data...")
try:
    for filename in os.listdir(TEST_FOLDER):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            filepath = os.path.join(TEST_FOLDER, filename)
            try:
                img = Image.open(filepath).convert("RGB")
                img = img.resize(IMG_SIZE)
                img_array = np.array(img)
                d_keys = filename[:-4]
                label = "Unknown"
                obj = ImageObject(img_array, label)
                test_data.append(obj)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
except FileNotFoundError:
    print(f"CRITICAL: Could not find CSV file at {CSV_PATH}")

print(f"Loaded {len(train_data)} train and {len(test_data)} test images.")

print("Saving data to 'dataset_cache.pkl'...")
with open("dataset_cache.pkl", "wb") as f:
    pickle.dump((train_data, test_data), f)
print("Data saved successfully!")
