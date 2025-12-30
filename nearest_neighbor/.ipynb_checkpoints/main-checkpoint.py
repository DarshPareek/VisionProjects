import pickle
import numpy as np


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


def load_cifar10():
    try:
        with open("../dataset_cache.pkl", "rb") as f:
            train_data, test_data = pickle.load(f)
        print("Success!")
        print(train_data[0])
        return train_data, test_data

    except FileNotFoundError:
        print("Cache file not found! Please run the setup script first.")
        return [], []


class NN_Classifier:
    def __init__(self, train_data) -> None:
        self.data = np.array([i.data.flatten() for i in train_data])
        self.reds = np.array([i.red.flatten() for i in train_data])
        self.greens = np.array([i.green.flatten() for i in train_data])
        self.blues = np.array([i.blue.flatten() for i in train_data])
        self.grays = np.array([i.grayscale.flatten() for i in train_data])
        self.labels = np.array([i.label for i in train_data])

    def __repr__(self) -> str:
        return f"Data: {len(self.data)}, Labels: {len(self.labels)}\nIndex 0: {self.data[0]}, label: {self.labels[0]}"

    def test(self, test_data, color, metric):
        # Basic grayscale tester with manhattan distance
        results = []
        test_matrix = np.array([i.grayscale.flatten() for i in test_data])
        test_labels = [i.label for i in test_data]
        print(
            f"Testing {len(test_data)} images against {len(self.grays)} training samples..."
        )
        for i, test_vec in enumerate(test_matrix):
            diffs = self.grays - test_vec
            abs_diffs = np.abs(diffs)
            distances = np.sum(abs_diffs, axis=1)
            min_index = np.argmin(distances)
            predicted_label = self.labels[min_index]
            results.append(predicted_label)
            if i % 100 == 0:
                print(
                    f"Processed {i}/{len(test_matrix)} | Pred: {predicted_label} | Acctual {test_labels[i]}"
                )
        return results


train_data = []
test_data = []
train_data, test_data = load_cifar10()

print(f"Train size: {len(train_data)}")
print(f"Test size: {len(test_data)}")
xy_train = train_data[:40000]
xy_test = train_data[40000:]
classifier = NN_Classifier(xy_train)
print(classifier)
results = classifier.test(xy_test, None, None)
print(results)
