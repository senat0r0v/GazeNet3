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
 - Location: GazeNet3/dataset/
 - Structure:
   - train/: Contains synthetic images for training, categorized into Center, Down, and Up.
   - test/: Contains synthetic images for testing, similarly categorized.
 - Usage: Provides a foundation for models to learn basic gaze direction features.

### Natural Dataset
The natural dataset captures real-world variations to improve model generalization.
 - Location: GazeNet3/natural_dataset/
 - Structure:
   - train/: Contains natural eye images for training.
   - test/: Contains natural eye images for testing.
 - Approximately 1,000 images in total for both training and testing.
 - Naming Convention: Images are named with a unique 8-character alphanumeric ID followed by the class label (e.g., eyes_0AQVX3JG_Down.jpg).
 - Usage: Enhances models' ability to generalize to diverse, real-world conditions.

### Real Eyes for Testing
This dataset is used exclusively for evaluating the inference performance of trained models on unseen real-eye images.
 - Location: GazeNet3/dataset/real_eyes_for_testing/
 - Structure:
   - Contains 10 images per class (Center, Down, Up).
 - Usage: Provides a benchmark to assess how well models perform on completely new data.

### UnityEyes Software
To generate the synthetic dataset, we utilized UnityEyes, a tool for rapidly synthesizing large amounts of variable eye region images for training data.

Summary of UnityEyes:
UnityEyes is a novel method that allows for the rapid synthesis of large volumes of eye region images with high variability. It combines a generative 3D model of the human eye region with a real-time rendering framework. The key features include:
 - Realistic Eye Region Modeling: Based on high-resolution 3D face scans, the model accurately represents the human eye region.
 - Anatomically Inspired Animations: Procedural geometry methods simulate realistic eyelid movements.
 - Complex Material Approximation: Real-time approximations for eyeball materials and structures enhance realism.
 - Gaze Variation: Capable of synthesizing images with a wide range of head poses and gaze angles, including extreme cases.
UnityEyes has proven effective for gaze estimation in challenging scenarios, even when the pupil is fully occluded. It enables the creation of diverse training datasets without the need for labor-intensive data collection and labeling.

Official Link: https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/
Authors: Erroll Wood, Tadas Baltrušaitis, Louis-Philippe Morency, Peter Robinson, and Andreas Bulling





