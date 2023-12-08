from django.conf import settings
import cv2
import os
import numpy as np
from PIL import Image
from io import BytesIO
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm
import matplotlib
from .models import ImageData, ModelWeights

matplotlib.use('Agg') 


def save_model(model):
    base_folder = "media/ModelWeights"
    base_filename = "fruitscan_model_weights_v"
    version = 1

    # Ensure the base folder exists, create it if it doesn't
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    while True:
        # Construct the filename with the current version number
        filename = f"{base_folder}/{base_filename}{version}.h5"

        # Check if this filename already exists
        if not os.path.exists(filename):
            # If the file does not exist, save the model
            model.save_weights(filename)
            break 

        # Increment the version number and try again
        version += 1
    
    return version

def train_model():
    # Variables for image
    width = 256
    height = 256
    channels = 3
    num_classes = 4
    epochs = 5

    # Load images and labels
    image_objects = ImageData.objects.all()
    images = []
    labels = []
    
    # First, count the total number of images
    total_images = image_objects.count()

    progress_bar = tqdm(total=total_images, desc="Loading Images", unit="image")
    
    for image_obj in image_objects:
        img_data = BytesIO(image_obj.image_data)
        img = Image.open(img_data)
        img = img.resize((width, height))
        img = np.array(img)
        if img.shape == (width, height, channels):
            img = img.astype(np.float32) / 255.0  # Normalization step
            images.append(img)
            labels.append(image_obj.label)

        progress_bar.update(1)
    progress_bar.close()

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Split dataset for training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=8
    )

    # Create model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, channels)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        #Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model and save accuracy for different epochs
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))

    # get training and validation accuracy values from the last epoch
    training_accuracy = history.history['accuracy'][-1]
    validation_accuracy = history.history['val_accuracy'][-1]
    #Convert to decimal values to fit db model
    training_accuracy_float = float(training_accuracy)
    validation_accuracy_float = float(validation_accuracy)

    # Save model to file
    model_version = save_model(model)

    # Use the validation set on the confusion matrix
    y_val_pred = model.predict(X_val)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)

    # Create confusion matrix
    conf_matrix_val = confusion_matrix(y_val, y_val_pred_classes)

    # Plot the confusion matrix for the validation set
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_val, annot=True, fmt='g')
    plt.title('Confusion Matrix (Validation Set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Define the directory for saving the confusion matrix
    conf_matrix_dir = os.path.join(settings.BASE_DIR, 'media/Performance/ConfusionMatrix')
    os.makedirs(conf_matrix_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Save the plot as an image file in the specified directory
    plt.savefig(os.path.join(conf_matrix_dir, f'confusion_matrix_v{model_version}.png'))
    print("Confusion matrix created.")

    # Plot and save the CNN model diagram
    cnn_plot_dir = os.path.join(settings.BASE_DIR, 'media/Performance/CnnModel')
    os.makedirs(cnn_plot_dir, exist_ok=True)  # Create directory if it doesn't exist

    tf.keras.utils.plot_model(
        model,
        to_file=os.path.join(cnn_plot_dir, f"cnn_model_v{model_version}.png"),
        show_shapes=True,
        show_dtype=False,
        show_layer_names=False,
        rankdir="TB", #LR or TB
        expand_nested=False,
        dpi=96,
        layer_range=None,
        show_layer_activations=True,
        show_trainable=False,
    )
    print("Model plotted.")
    print("Model trained successfully.")
    
    # Add model and files connected to model in DB
    model_path = f"ModelWeights/fruitscan_model_weights_v{model_version}.h5"
    confusion_matrix_path = f"Performance/ConfusionMatrix/confusion_matrix_v{model_version}.png"

    new_db_entry = ModelWeights(version=model_version, path=model_path, confusion_matrix=confusion_matrix_path, train_accuracy= training_accuracy_float, val_accuracy=validation_accuracy_float)
    new_db_entry.save()
    print(f'Model {model_version} added to database.')
