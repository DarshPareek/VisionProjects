import pickle
import numpy as np


class ImageObject:
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __repr__(self):
        return f"<ImageObject file={self.label} shape={self.data.shape}>"


def load_cifar10():
    try:
        with open("./dataset_cache.pkl", "rb") as f:
            train_data, test_data = pickle.load(f)
        print("Success!")
        print(train_data[0])
        return train_data, test_data

    except FileNotFoundError:
        print("Cache file not found! Please run the setup script first.")
        return [], []


"""
import scripts
from scripts import ImageObject

train_data = []
test_data = []
train_data, test_data = scripts.load_cifar10()

print(f"Train size: {len(train_data)}")
print(f"Test size: {len(test_data)}")

"""
