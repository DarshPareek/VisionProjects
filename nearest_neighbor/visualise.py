import pickle
import numpy as np
import base64
import io
from PIL import Image
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool


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
        self.data = []
        self.reds = []
        self.greens = []
        self.blues = []
        self.grays = []
        self.labels = []
        for i in train_data:
            self.data.append(i.data)
            self.reds.append(i.red)
            self.greens.append(i.green)
            self.blues.append(i.blue)
            self.grays.append(i.grayscale)
            self.labels.append(i.label)

    def __repr__(self) -> str:
        return f"Data: {len(self.data)}, Labels: {len(self.labels)}\nIndex 0: {self.data[0]}, label: {self.labels[0]}"


def array_to_html_b64(img_array):
    img_array = img_array.astype(np.uint8)
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    b64_string = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64_string}"


train_data = []
test_data = []
train_data, test_data = load_cifar10()

print(f"Train size: {len(train_data)}")
print(f"Test size: {len(test_data)}")
classifier = NN_Classifier(train_data)
LIMIT = 500  # Adjust as needed
print(f"Processing first {LIMIT} images...")
images = [obj for obj in train_data[:LIMIT]]
labels = [obj.label for obj in train_data[:LIMIT]]
points = []
images_b64 = []
x_val, y_val = [], []
for i in range(len(images)):
    imgd = images[i].data
    imgr = images[i].red
    imgg = images[i].green
    imgb = images[i].blue
    imggr = images[i].grayscale
    points.append(
        (
            np.sum(imgr / 255)
            + np.sum(imgb / 255)
            + np.sum(imgg / 255)
            + np.sum(imggr / 255),
            labels[i],
        )
    )
    x_val.append(
        np.sum(imgr / 255)
        + np.sum(imgb / 255)
        + np.sum(imgg / 255)
        + np.sum(imggr / 255)
    )
    y_val.append(labels[i])
    images_b64.append(array_to_html_b64(imgd))

img_src = array_to_html_b64(images[0].data)

unique_labels = sorted(list(set(y_val)))
source = ColumnDataSource(data=dict(x=x_val, y=y_val, label=labels, image=images_b64))
p = figure(
    title="CIFAR-10: Images vs Brightness",
    x_axis_label="Brightness",
    y_axis_label="Categories",
    y_range=unique_labels,
    tools="pan,wheel_zoom,box_zoom,reset",
    width=800,
    height=600,
)
points_renderer = p.scatter(
    "x",
    "y",
    source=source,
    marker="circle",
    fill_alpha=0.6,
    line_color=None,
)
hover = HoverTool(
    tooltips="""
    <div style="background: white; padding: 10px; border: 1px solid #ccc;">
        <div>
            <img
                src="@image" alt="@label"
                style="width:128px; height:128px; image-rendering: pixelated;
                border="2"
            ></img>
        </div>
        <div>
            <span style="font-size: 12px; color: #666;">Brightness: @x{0.0}</span>
            <br>
            <span style="font-size: 12px; color: #666;">Label: @label</span>
        </div>
    </div>
    """
)

p.add_tools(hover)

output_file("cifar_scatter.html")
print("Opening plot...")
show(p)
