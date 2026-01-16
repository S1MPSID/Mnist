MNIST Digit Classification using ANN and CNN
📌 Project Overview

This project implements and compares different deep learning models for handwritten digit classification using the MNIST dataset. The work progresses from a simple Artificial Neural Network (ANN) baseline to a more advanced Convolutional Neural Network (CNN), and finally to a CNN with Dropout to reduce overfitting and improve generalization.

The models are developed and evaluated using TensorFlow and Keras in Google Colab.

🎯 Objectives

To understand and implement ANN and CNN architectures for image classification

To compare model performance on the MNIST dataset

To analyze overfitting and apply dropout regularization

To visualize and interpret training and validation performance

📂 Project Structure
├── ann+cnn.py           # ANN and CNN baseline implementation
├── cnn_dropout.py       # CNN with Dropout and performance analysis
└── README.md            # Project documentation

🗂 Dataset

Dataset: MNIST Handwritten Digits

Image size: 28 × 28 (grayscale)

Classes: Digits 0–9

Training samples: 60,000

Test samples: 10,000

The dataset is loaded directly using tensorflow.keras.datasets.mnist.

🧠 Models Implemented
1️⃣ Artificial Neural Network (ANN)

Input images are flattened into a 1D vector

Fully connected dense layers with ReLU activation

Softmax output layer for multi-class classification

Purpose:
Serves as a baseline model for comparison.

2️⃣ Convolutional Neural Network (CNN)

Convolution layers for feature extraction

Max pooling layers for dimensionality reduction

Dense layers for classification

Purpose:
Preserves spatial features in images and improves accuracy compared to ANN.

3️⃣ CNN with Dropout

Dropout layers added after convolution and dense layers

Helps reduce overfitting by randomly deactivating neurons during training

Purpose:
Improves model generalization and stabilizes validation performance.

⚙️ Implementation Details

Framework: TensorFlow (Keras API)

Optimizer: Adam

Loss Function: Sparse Categorical Crossentropy

Epochs: 5

Validation Split: 10%

Models are trained and evaluated on unseen test data to measure performance.

📊 Results Summary
Model	Test Accuracy
ANN	~95–97%
CNN	~98–99%
CNN + Dropout	~97–98%

Training vs validation accuracy plots are generated to analyze overfitting behavior in the dropout model.

🔍 What the Code Does

Loads and preprocesses MNIST images

Trains ANN and CNN models to learn digit patterns

Evaluates model performance on test data

Visualizes predictions for selected test images

Saves trained CNN models for reuse

Analyzes generalization using dropout and accuracy curves

The models recognize handwritten digits from the MNIST dataset, not real-time user input.

📈 Visualizations

Sample MNIST images

Model architecture summaries

Training and validation accuracy graphs

Predicted vs true label comparison for test samples

🚀 Future Scope

Deployment of the trained CNN model in a web application

Extension to real-world handwritten digit inputs

Experimentation with deeper architectures and hyperparameter tuning

🏁 Conclusion

This project demonstrates the effectiveness of CNNs over traditional ANN models for image classification tasks. It also highlights the importance of regularization techniques such as dropout to control overfitting and improve model robustness.
