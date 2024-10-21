# GazeNet3: Robust Gaze Direction Prediction
Welcome to GazeNet3, a comprehensive project dedicated to developing, training, and deploying neural network models for accurate gaze direction prediction. This README provides an overview of the project's structure, datasets, training scripts, model performances, Unity integration for real-time gaze detection.

![inference_results](https://github.com/user-attachments/assets/65d494f4-ae8b-4009-97fd-3181e041ca08)

## Table of Contents
1. Introduction
2. Project Overview
3. Datasets
 - Synthetic Dataset
 - Natural Dataset
 - Real Eyes for Testing
 - UnityEyes Software
4. Training Environment
5. Training Scripts
6. Model Performances
7. Unity Integration
 - GridColorAssigner.cs
 - IOSGazeController.cs
8. Collaborators and Acknowledgments
9. Summary and Future Work
10. License

## Introduction
GazeNet3 aims to create robust neural network models capable of predicting gaze direction with high accuracy. By leveraging both synthetic and natural datasets, the project trains various architectures to generalize well to real-world scenarios. Integration with Unity allows for real-time gaze prediction, enhancing applications in user interaction, accessibility, and more.

## Project Overview
GazeNet3 encompasses the following key components:
 - Datasets: A combination of synthetic and natural eye images labeled with gaze directions (Up, Down, Center).
 - Training Scripts: Python scripts for training different neural network architectures, including AlexNet, GazeNet3, MobileNetV2, and ResNet-152.
 - Models: Trained models saved in both PyTorch (.pth) and ONNX (.onnx) formats for flexibility in deployment.
 - Unity Integration: Scripts to collect natural eye images and implement real-time gaze prediction within a Unity application.

## Datasets
### Synthetic Dataset
The synthetic dataset is used to train and evaluate models on controlled, consistent data to establish baseline performances.

Location: GazeNet3/dataset/
Structure:
train/: Contains synthetic images for training, categorized into Center, Down, and Up.
test/: Contains synthetic images for testing, similarly categorized.
Usage: Provides a foundation for models to learn basic gaze direction features.
