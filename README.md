# Severstal Steel Defect Detection

## Overview
This project aims to detect defects in Severstal steel images using a deep learning model. The dataset consists of images with various types of defects, and the project includes data preprocessing, model training, evaluation, and visualization of results.

## Project Structure
- **data/**: Contains image datasets for training, validation, and testing.
  - **train_images/**: Images used for training the model.
  - **validation_images/**: Images used for validating the model.
  - **test_images/**: Images used for testing the model.

- **annotations/**: XML annotations for training and validation.
  - **train/**: XML files for training images.
  - **validation/**: XML files for validation images.

- **scripts/**: Python scripts for preprocessing, training, and evaluation.
  - **PP.py**: Data preprocessing script that prepares images and annotations.
  - **MT.py**: Model training script that trains the model on the dataset.
  - **MS.py**: Model saving script that saves the trained model in Keras format.
  - **IN.py**: Model inference script for making predictions on new images.
  - **CAM_IN.py**: Model evaluation script that generates performance metrics and visualizations.

- **models/**: Contains saved models.
  - **model.keras**: Trained model in Keras format.

- **results/**: Contains results from model training and evaluation.
  - **training_curves.png**: Graph showing training and validation accuracy and loss curves.
  - **predicted_output.png**: Example image with predicted defect annotations.
  - **balanced_dataset_graph.png**: Graph showing the balanced distribution of the dataset.
  - **model_summary.txt**: Summary of the trained model.

## Model Architecture
The model is based on the ResNet50 architecture, a popular deep learning model known for its performance in image classification tasks. It includes:
- **Base Model:** ResNet50 with weights pre-trained on ImageNet.
- **Custom Layers:** Global Average Pooling, Dense layers with ReLU activation, and a final Dense layer with Softmax activation for classification.
- **Frozen Layers:** The layers of ResNet50 are frozen during training to leverage pre-trained features.

## Results
### Accuracy and Loss Curves
![Training and Validation Curves](results/training_curves.png)

This graph illustrates the training and validation accuracy and loss over the epochs.

### Predicted Output
![Predicted Output Image](results/predicted_output.png)

An example of an image with defect annotations predicted by the model.

### Dataset Balanced Graph
![Balanced Dataset Graph](results/balanced_dataset_graph.png)

This graph shows the balanced distribution of the dataset classes.

## Installation
Ensure you have the following Python packages installed:
- TensorFlow
- NumPy
- pandas
- matplotlib
- scikit-learn

You can install these packages using `pip`:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
