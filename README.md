# Diabetic Retinopathy Detection with CNN and Gradio

This project demonstrates the use of a Convolutional Neural Network (CNN) to classify images of retina as having different severities of diabetic retinopathy. The project utilizes TensorFlow for model training and Gradio for creating a user-friendly interface for predictions.

## Table of Contents

1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Data Preparation](#data-preparation)
5. [Training the Model](#training-the-model)
6. [Evaluation](#evaluation)
7. [Creating a Gradio Interface](#creating-a-gradio-interface)
8. [Running the Application](#running-the-application)
9. [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Diabetic-Retinopathy-Detection.git
    cd Diabetic-Retinopathy-Detection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The dataset used for this project is the [Diabetic Retinopathy: 224x224 Gaussian Filtered](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered?select=gaussian_filtered_images) dataset from Kaggle.

1. Download the dataset from Kaggle and extract it into a directory named `data`.

    ```bash
    kaggle datasets download -d sovitrath/diabetic-retinopathy-224x224-gaussian-filtered
    unzip diabetic-retinopathy-224x224-gaussian-filtered.zip -d data
    ```

## Model Architecture

The model used in this project is a Convolutional Neural Network (CNN) based on ResNet50 with additional layers for classification.

## Data Preparation

Data preparation includes transforming the images and splitting the dataset into training, validation, and test sets.

## Training the Model

Training the model involves several epochs of forward and backward propagation using a defined loss function and optimizer.

## Evaluation

Evaluation includes checking the model's performance on the validation set and using metrics such as accuracy, confusion matrix, and classification report.

## Creating a Gradio Interface

We use Gradio to create an interface for users to upload images and get predictions.

## Running the Application

Run the following command to start the Gradio interface:
```bash
python main.py

