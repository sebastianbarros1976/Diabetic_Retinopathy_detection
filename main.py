from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the path to the training data
train_folder_path = '/content/drive/My Drive/Colab Notebooks/Train'

# Initialize lists to store image paths and corresponding labels
train = []
label = []

# Iterate over each class directory in the train folder
for category in os.listdir(train_folder_path):
    category_path = os.path.join(train_folder_path, category)
    if os.path.isdir(category_path):
        for image in os.listdir(category_path):
            image_path = os.path.join(category_path, image)
            train.append(image_path)
            label.append(category)

# Create a DataFrame
retina_df = pd.DataFrame({'Image': train, 'Labels': label})

# Shuffle and split the data
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

retina_df = shuffle(retina_df)
train_df, test_df = train_test_split(retina_df, test_size=0.2)

# Data augmentation and generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.15
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='Image',
    y_col='Labels',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='Image',
    y_col='Labels',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='Image',
    y_col='Labels',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model, Sequential

# Load the ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Build the model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('/content/drive/My Drive/Colab Notebooks/retinopathy_model.h5', save_best_only=True, monitor='val_loss')

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
    callbacks=[early_stopping, model_checkpoint]
)

# Save the final model
model.save('/content/drive/My Drive/Colab Notebooks/retinopathy_model_final.h5')

from tensorflow.keras.models import load_model

# Load the trained model
retinopathy_model = load_model('/content/drive/My Drive/Colab Notebooks/retinopathy_model.h5')

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluate the model
test_loss, test_acc = retinopathy_model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Get predictions
y_pred = retinopathy_model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Classification report
print("Classification Report:\n", classification_report(y_true, y_pred_classes))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

import cv2

# Define class names
class_names = {0: 'No_DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative_DR'}

# Define a function to predict the class of a single image
def predict_retinopathy(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Check if the image was loaded correctly
    if image is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")
    # Preprocess the image
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    # Make a prediction
    prediction = retinopathy_model.predict(image).flatten()
    # Format the result
    result = {class_names[i]: float(prediction[i]) for i in range(5)}
    return result

# Path to the test image
test_image_path = "/content/drive/My Drive/Colab Notebooks/TestImage.jpg"  # Ensure this path is correct

# Get the prediction
try:
    prediction = predict_retinopathy(test_image_path)
    print("Prediction for the selected test image:", prediction)
except ValueError as e:
    print(e)

import gradio as gr

# Define the prediction function for Gradio interface
def predict_retinopathy_gradio(image):
    # Preprocess the image
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    # Make a prediction
    prediction = retinopathy_model.predict(image).flatten()
    # Format the result
    return {class_names[i]: float(prediction[i]) for i in range(5)}

# Define the input and output interfaces
image_input = gr.Image(shape=(224, 224))
label_output = gr.Label(num_top_classes=5)

# Create the Gradio interface
interface = gr.Interface(fn=predict_retinopathy_gradio, inputs=image_input, outputs=label_output,
                         title="Diabetic Retinopathy Detection",
                         description="Upload a retina image to classify its severity of diabetic retinopathy.")

# Launch the Gradio interface
interface.launch(share=True)
