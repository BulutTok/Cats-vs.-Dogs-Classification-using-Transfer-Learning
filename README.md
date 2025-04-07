# Cats vs. Dogs Classification using Transfer Learning

This repository contains an end-to-end example of using transfer learning for a computer vision task—classifying images of cats and dogs. The project leverages TensorFlow and the pre-trained InceptionV3 model to build a binary image classifier. It demonstrates how to download and prepare data, create training and validation sets, augment image data, build and fine-tune a neural network, and finally make predictions on new images.

---

## Table of Contents

- [Overview](#overview)
- [License](#license)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Code Walkthrough](#code-walkthrough)
- [Results and Evaluation](#results-and-evaluation)
- [Acknowledgments](#acknowledgments)

---

## Overview

In this project, we:
- **Download and extract the dataset:** The cats vs. dogs dataset is downloaded from Microsoft and unzipped.
- **Prepare the data:** The dataset is split into training and testing sets for both cats and dogs. Files that are zero-length are skipped.
- **Augment the images:** The training data is augmented using various transformations to improve model generalization.
- **Load a pre-trained model:** We use the InceptionV3 model (without the top layers) with pre-trained weights. All layers are frozen to retain learned features.
- **Build and compile a new model:** New fully connected layers are added on top of InceptionV3 for binary classification.
- **Train the model:** The model is trained with the augmented data, and training progress is plotted.
- **Make predictions:** Finally, the model is used to classify new images of cats and dogs.

---

## License

This project is licensed under the Apache License, Version 2.0.  
You may not use this file except in compliance with the License.  
You can obtain a copy of the License at:  
[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/cats-vs-dogs-transfer-learning.git
   cd cats-vs-dogs-transfer-learning
   ```

2. **Install dependencies:**  
   This project requires TensorFlow 2.x, Matplotlib, and other standard Python libraries. Install the dependencies using pip:

   ```bash
   pip install tensorflow matplotlib
   ```

3. **(Optional) Use Google Colab:**  
   The code includes a `%tensorflow_version 2.x` magic command for Colab users. If you are using Colab, simply open the notebook.

---

## Usage

1. **Download and extract the dataset:**  
   The script downloads the cats vs. dogs dataset from Microsoft, extracts it to a temporary directory, and prints the number of images in each category.

2. **Prepare the data:**  
   The dataset is split into training and testing directories. Approximately 90% of the images are allocated for training and 10% for testing. Files with zero length are ignored.

3. **Data augmentation:**  
   The `ImageDataGenerator` is used to augment training images with rotation, shifting, shearing, zooming, and horizontal flipping.

4. **Model building and training:**  
   A pre-trained InceptionV3 model is loaded without its top layers. New dense layers are added for binary classification (cat vs. dog), and the model is compiled with the RMSprop optimizer and binary crossentropy loss. The model is then trained on the augmented data for 20 epochs.

5. **Evaluation and prediction:**  
   Training progress (accuracy and loss) is plotted, and the model can be used to predict new images. Uploaded images are processed and classified as either cat or dog.

To run the entire process, simply execute the script or Jupyter Notebook in your preferred Python environment.

---

## Project Structure

```
.
├── README.md                 # This file
├── cats_vs_dogs.py           # Main script (or Notebook) with the code
└── LICENSE                   # Apache License, Version 2.0
```

---

## Code Walkthrough

### 1. License Header and Environment Setup

The file begins with the Apache License header and attempts to set TensorFlow to version 2.x (useful for Google Colab).

```python
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    %tensorflow_version 2.x
except Exception:
    pass
```

### 2. Data Download and Extraction

The dataset is downloaded from Microsoft and extracted into a temporary directory.

```python
import urllib.request
import os
import zipfile

data_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip"
data_file_name = "catsdogs.zip"
download_dir = '/tmp/'
urllib.request.urlretrieve(data_url, data_file_name)
zip_ref = zipfile.ZipFile(data_file_name, 'r')
zip_ref.extractall(download_dir)
zip_ref.close()
```

### 3. Data Preparation

Directories for training and testing are created, and the images are split using a custom function that randomly assigns files to each set. Files with zero length are skipped.

```python
from shutil import copyfile
import random

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = len(files) - training_length
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[:training_length]
    testing_set = shuffled_set[training_length:]

    for filename in training_set:
        copyfile(SOURCE + filename, TRAINING + filename)
    for filename in testing_set:
        copyfile(SOURCE + filename, TESTING + filename)

# Define source and target directories
CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

split_size = 0.9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)
```

### 4. Data Augmentation and Generators

The training data is augmented using various transformations. A validation generator is also created for the testing data.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAINING_DIR = "/tmp/cats-v-dogs/training/"
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size=100,
    class_mode='binary',
    target_size=(150, 150)
)

VALIDATION_DIR = "/tmp/cats-v-dogs/testing/"
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size=100,
    class_mode='binary',
    target_size=(150, 150)
)
```

### 5. Building and Compiling the Model

A pre-trained InceptionV3 model (without the top layer) is loaded with pre-trained weights. The layers are frozen, and new fully connected layers are added for classification.

```python
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import RMSprop

weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file = "inception_v3.h5"
urllib.request.urlretrieve(weights_url, weights_file)

pre_trained_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None)
pre_trained_model.load_weights(weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

# Add custom layers on top of the pre-trained model
x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(optimizer=RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])
```

### 6. Model Training and Evaluation

The model is trained for 20 epochs using the training generator and evaluated with the validation generator. Training progress is visualized with accuracy and loss plots.

```python
history = model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    verbose=1
)

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()
```

### 7. Making Predictions on New Images

The script also includes code for uploading images and making predictions on whether the image is of a cat or a dog.

```python
import numpy as np
from google.colab import files
from tensorflow.keras.preprocessing import image

uploaded = files.upload()
for fn in uploaded.keys():
    path = '/content/' + fn
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor)
    print(classes)
    if classes[0] > 0.5:
        print(fn + " is a dog")
    else:
        print(fn + " is a cat")
```

---

## Results and Evaluation

After training, the model's performance is evaluated on the validation dataset. Training and validation accuracy curves are plotted to help diagnose model behavior such as overfitting. Experiment with the parameters (data augmentation, number of epochs, learning rate, etc.) to further optimize performance.

---

## Acknowledgments

- **Apache License, Version 2.0:** This project is licensed under the Apache License.
- **TensorFlow:** For the deep learning framework and pre-trained models.
- **InceptionV3:** For providing a powerful architecture for transfer learning.
- **Microsoft and Kaggle:** For the cats vs. dogs dataset.
