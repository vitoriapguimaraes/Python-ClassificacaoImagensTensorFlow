# Image Classification with Convolutional Neural Networks (CNNs) using TensorFlow

# (1) Importing the necessary libraries
from platform import python_version
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

print("Python version:", python_version())
print("TensorFlow version:", tf.__version__)

print("(1) Importing the necessary libraries - Status: done\n")

# (2) Loading and initial exploration of data

(images_training, labels_training), (images_testing, labels_testing) = datasets.cifar10.load_data()

classification_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print("(2) Loading and initial exploration of data - Status: done\n")

# (3) Image pre-processing

images_training = images_training / 255.0
images_testing = images_testing / 255.0

def visualize_images(images, labels):
    plt.figure(figsize = (10,10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap = plt.cm.binary)
        plt.xlabel(classification_names[labels[i][0]])
    plt.show()

visualize_images(images_training, labels_training)

print("(3) Image pre-processing - Status: done\n")

# (4) Building and Training the Model

classification_model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

print("### Classification model Summary ###")
print(classification_model.summary())

classification_model.compile(optimizer = 'adam', 
                   loss = 'sparse_categorical_crossentropy', 
                   metrics = ['accuracy'])

print("Training the model...")

history = classification_model.fit(images_training, 
                         labels_training, 
                         epochs = 10, 
                         validation_data = (images_testing, labels_testing))

print("Training the model...ok")
print("(4) Building and Training the Model - Status: done\n")

# (5) Performance evaluation

erro_testing, acc_testing = classification_model.evaluate(images_testing, labels_testing, verbose = 2)     

print(f"Accuracy with Test Data: {acc_testing}")

print("(5) Performance evaluation - Status: done\n")

# (6) Model deployment and predictions

data_path = "data/image-entry.jpg"

image_input = Image.open(data_path)
image_input = image_input.resize((32, 32))
image_input_array = np.array(image_input) / 255.0

print(f"Imported {data_path}")

image_input = np.expand_dims(image_to_classify_array, axis=0)
predictions = classification_model.predict(image_input)

classification_answer = np.argmax(predictions)
classification_name_answer = classification_names[classification_answer]

print(f"The input image ({data_path}) was classified as '{classification_name_answer}'.")
