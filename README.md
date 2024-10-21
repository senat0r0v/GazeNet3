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

## Training Environment
 - Operating System: Ubuntu running on Windows Subsystem for Linux (WSL)
 - Reason: WSL was chosen to resolve module conflicts encountered on Windows, providing a Linux-like environment within Windows for seamless dependency management and execution of training scripts.

## Training Scripts
The trainers directory contains Python scripts for training different neural network architectures on specified datasets.

Overview of Training Scripts:
1. AlexNet Trainers:
   - alexnet_trainer_synthetic.py: Trains AlexNet on the synthetic dataset.
   - mixed_alexnet_trainer.py: Trains AlexNet on a mixed dataset (synthetic + natural).
   - natural_alexnet_trainer.py: Fine-tunes AlexNet exclusively on the natural dataset.
2. GazeNet3 Trainer:
   - gazenet3_trainer_mixed.py: Trains the custom GazeNet3 model on a mixed dataset.
3. MobileNetV2 Trainers:
   - mobilenetv2_trainer.py: Trains MobileNetV2 on the synthetic dataset.
   - natural_mobilenetv2_trainer.py: Fine-tunes MobileNetV2 on the natural dataset with data augmentation.
4. ResNet-152 Trainers:
   - resnet_trainer_synthetic.py: Trains ResNet-152 on the synthetic dataset.
   - natural_resnet_trainer.py: Fine-tunes ResNet-152 on the natural dataset.

Common Features Across Scripts:
 - Data Loading:
   - Utilizes PyTorch's ImageFolder and DataLoader for efficient data handling.
   - Applies necessary transformations like resizing, normalization, and augmentation.
 - Model Modification:
   - Pre-trained models are modified to output three classes corresponding to gaze directions.
 - Training Configurations:
   - Optimizers: Generally use AdamW for effective optimization.
   - Learning Rates: Adjusted based on dataset complexity.
   - Batch Sizes: Vary to balance GPU utilization and training stability.
   - Epochs: Set to ensure sufficient training without overfitting.
 - Performance Enhancements:
   - Mixed Precision Training: Implements torch.cuda.amp for faster training and reduced memory usage.
   - Optimized Data Loading: Uses pin_memory=True and an optimal number of workers.

## Model Performances

The performance of each trained model was evaluated on the real_eyes_for_testing dataset. Below is a summary of the results:

| Model Name            | Up (%) | Down (%) | Center (%) | Overall Accuracy (%) |
|-----------------------|--------|----------|------------|----------------------|
| alexnet               | 0.0    | 100.0    | 0.0        | 33.3                 |
| gazenet3              | 0.0    | 100.0    | 40.0       | 46.7                 |
| mixed_alexnet         | 20.0   | 100.0    | 30.0       | 50.0                 |
| mobilenetv2           | 0.0    | 100.0    | 0.0        | 33.3                 |
| natural_alexnet       | 20.0   | 20.0     | 100.0      | 46.7                 |
| natural_mobilenetv2   | 0.0    | 100.0    | 20.0       | 40.0                 |
| natural_resnet        | 0.0    | 0.0      | 100.0      | 33.3                 |
| resnet                | 0.0    | 100.0    | 0.0        | 33.3                 |

Observations:
 - Mixed AlexNet achieved the highest overall accuracy at 50.0%, indicating better generalization by leveraging both synthetic and natural datasets.
 - Models trained exclusively on natural data performed better on the Center class.
 - The Down class consistently showed high accuracy across most models, suggesting that downward gaze is easier for models to predict accurately.
 - There is room for improvement in predicting the Up and Center classes.

## Unity Integration
The Unity project includes scripts for data collection and real-time gaze prediction, enhancing the practical application of the trained models.
##### GridColorAssigner.cs
Purpose: Manages the visualization grid within the Unity scene, handles user interactions (cube taps), and facilitates the capture of eye images for the natural dataset.

Key Functionalities:
 - Grid Generation:
   - Creates a grid of colored cubes representing different gaze regions.
   - Assigns colors based on spatial regions to indicate gaze directions.
 - Cube Interaction:
   - Detects when a user taps a white cube on the grid.
   - Identifies the region corresponding to the tapped cube.
 - Image Capture:
   - Captures a photo of the user's eye using the device's camera upon cube tap.
   - Saves the captured image with a unique filename indicating the region.

Usage:
 - Used to collect natural eye images labeled with gaze direction, contributing to the natural dataset.
 - Collaborators: The natural dataset was gathered with the help of fellow students using this script.

##### IOSGazeController.cs
Purpose: Handles live gaze direction prediction by processing camera input, extracting eye regions, and utilizing trained models to determine gaze direction in real-time.

Key Functionalities:
 - Camera Handling:
   - Accesses the device's front-facing camera with optimized resolution settings.
   - Ensures proper camera permissions and initialization.
 - Landmark Detection:
   - Uses DlibFaceLandmarkDetector to detect facial landmarks.
   - Extracts the right eye region based on detected landmarks.
 - Model Integration:
   - Loads the ONNX model using Unity's Barracuda library.
   - Ensures proper camera permissions and initialization.
 - Prediction Averaging:
   - Maintains a buffer of recent predictions.
   - Computes the mode of predictions over a defined window to stabilize gaze direction outputs.
 - Visualization:
   - Updates the grid to reflect the predicted gaze direction.
   - Displays the extracted eye image and logs for user feedback.

Usage:
 - Implements real-time gaze prediction in the Unity application, enabling interactive experiences based on user's gaze.

![Animation](https://github.com/user-attachments/assets/59033247-3528-409a-8bda-7bae04821d5b)

## Collaborators and Acknowledgments
This project was made possible with the support and contributions of several individuals:
 - Collaborators: Max Choi, Sebastian Garcia, Alex Gulewich, Alisa Ho, Xav Laugo, Preetham Mukundan, Spencer Paynter-Repay, Levi Spevakow, and Henry Williams

These collaborators were students in Mobile Development 4 at Vancouver Film School in the program Programming for Games, Web & Mobile. They contributed by participating in gathering the natural dataset using the GridColorAssigner script and provided valuable ideas and feedback throughout the project.

 - Instructor and Mentor: Amir Jahanlou

Amir was an incredible instructor and mentor who guided me through this project, offering insights and encouragement that were instrumental in its success.

## Summary and Future Work
Achievements:
 - Developed multiple models with varying architectures trained on synthetic, natural, and mixed datasets.
 - Integrated models into Unity for real-time gaze prediction.
 - Achieved the highest overall accuracy with the Mixed AlexNet model.
 - Utilized UnityEyes software to generate a diverse synthetic dataset.
 - Gathered a natural dataset with the help of collaborators, enhancing the models' real-world applicability.

Recommendations for Future Enhancements:
1. Enhanced Data Collection:
   - Collect pictures in the millions using an interactive approach, such as a game that encourages user participation.
   - Implement cloud-based storage for efficient data gathering and management.
   - Ensure real-time face detection and eye extraction using tools like DlibFaceLandmarkDetector (available on the Unity Asset Store for $40 USD as of writing).
2. Improve Data Diversity:
   - Increase the number of natural eye images, especially for the Up class.
   - Introduce variations in lighting, angles, and environmental conditions.
3. Advanced Data Augmentation:
   - Apply techniques like random occlusions, noise addition, and geometric transformations to simulate real-world challenges.
4. Hyperparameter Optimization:
   - Experiment with different learning rates, batch sizes, and optimizers to find the optimal training configurations.
5. Model Ensemble:
   - Combine predictions from multiple models to improve overall accuracy and robustness.
6. Explore New Architectures:
   - Test advanced models like EfficientNet or Vision Transformers (ViT) for potential performance gains.
7. Cross-Validation:
   - Implement k-fold cross-validation to assess model generalization more robustly.

Future Work:
 - Performance Optimization:
   - Optimize Unity scripts and model inference for better real-time performance on various devices.
 - User Interface Improvements:
   - Enhance the Unity application UI for an improved user experience, making it more intuitive and engaging.
 - Deployment:
   - Package the application for broader testing, gather user feedback, and iterate based on insights.
  
## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

We welcome contributions to this project. If you're interested in collaborating, please feel free to submit issues or pull requests. Your feedback and involvement are highly appreciated!

Note: For developers interested in using DlibFaceLandmarkDetector for face and eye detection, it is available on the Unity Asset Store for $40 USD (as of writing). This tool can significantly enhance data collection processes by automating face detection and eye extraction.

















