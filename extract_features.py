from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import os
import re

# Load the VGG16 model
base_model = VGG16(weights='imagenet')

# Choose the intermediate layer from which you want to extract features
intermediate_layer_name = 'fc2'

# Create a model that includes the intermediate layer
intermediate_layer_model = Model(
    inputs=base_model.input, outputs=base_model.get_layer(intermediate_layer_name).output)

# Function to extract features from a folder of images


def extract_features_from_folder(folder_path, model):
    features = []
    filenames = []

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = image.load_img(img_path, target_size=(
            224, 224))  # Adjust size as needed
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Extract features from the intermediate layer
        intermediate_features = model.predict(img_array)

        features.append(intermediate_features.flatten())
        filenames.append(re.sub(r'\d+', '', filename.split('.')[0]))

    return np.array(features), filenames


# Define paths to your training and validation folders
train_folder = 'simpsons/Train'
validation_folder = 'simpsons/Valid'

# Extract features from the training and validation folders
train_features, train_filenames = extract_features_from_folder(
    train_folder, intermediate_layer_model)
validation_features, validation_filenames = extract_features_from_folder(
    validation_folder, intermediate_layer_model)

# Save the features and filenames to text files
np.savetxt('Features/v1/train_features.txt', train_features)
np.savetxt('Features/v1/validation_features.txt', validation_features)
np.savetxt('Features/v1/train_labels.txt', np.array(train_filenames), fmt='%s')
np.savetxt('Features/v1/validation_labels.txt', np.array(
    validation_filenames), fmt='%s')
