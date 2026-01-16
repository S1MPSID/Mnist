🧠 MNIST DIGIT CLASSIFICATION USING ANN AND CNN

🔍 PROJECT OVERVIEW

A deep learning project that implements and compares Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) for handwritten digit classification using the MNIST dataset, with additional analysis using Dropout to reduce overfitting.

The project is implemented using TensorFlow (Keras) in Google Colab and focuses on understanding model architecture, performance comparison, and generalization behavior.

🎯 PROJECT OBJECTIVES

Implement an ANN baseline model for digit classification

Implement a CNN to capture spatial features in images

Compare ANN and CNN performance

Apply dropout regularization to reduce overfitting

Analyze training and validation accuracy trends

🗂 DATASET DETAILS

Dataset Name: MNIST Handwritten Digits

Image Size: 28 × 28 (grayscale)

Number of Classes: 10 (Digits 0–9)

Training Samples: 60,000

Testing Samples: 10,000

The dataset is loaded directly using:

tensorflow.keras.datasets.mnist

🏗 PROJECT STRUCTURE

├── ann+cnn.py        # ANN and CNN baseline implementation

├── cnn_dropout.py    # CNN with Dropout and performance analysis

├── README.md         # Project documentation

🧠 MODELS IMPLEMENTED
🔹 1. ARTIFICIAL NEURAL NETWORK (ANN)

Input images flattened into 1D feature vectors

Fully connected dense layers

ReLU activation for hidden layers

Softmax activation for output layer

Purpose:
Serves as a baseline model for comparison.

🔹 2. CONVOLUTIONAL NEURAL NETWORK (CNN)

Convolution layers for feature extraction

Max pooling layers for dimensionality reduction

Dense layers for final classification

Purpose:
Preserves spatial information in images and significantly improves accuracy compared to ANN.

🔹 3. CNN WITH DROPOUT (IMPROVED MODEL)

Dropout layers added after convolutional and dense layers

Randomly disables neurons during training

Purpose:
Reduces overfitting and improves model generalization.

⚙️ TECHNICAL DETAILS

Framework: TensorFlow (Keras API)

Optimizer: Adam

Loss Function: Sparse Categorical Crossentropy

Epochs: 5

Validation Split: 10%

📊 RESULTS AND PERFORMANCE
Model	Test Accuracy
ANN	~95–97%
CNN	~98–99%
CNN + Dropout	~97–98%

Training vs validation accuracy plots are generated to analyze overfitting behavior in the dropout model.

🔍 WHAT THE CODE DOES

Loads and preprocesses MNIST images

Trains ANN and CNN models on training data

Evaluates performance on unseen test data

Compares model accuracy across architectures

Visualizes training and validation accuracy trends

Analyzes the impact of dropout regularization

⚠️ Note:
The models classify digits from the MNIST dataset and do not take real-time user input.

🚀 FUTURE IMPROVEMENTS

Deploy the trained CNN model using a web interface

Extend the model to handle real-world handwritten digit images

Experiment with deeper architectures and hyperparameter tuning

🏁 CONCLUSION

This project demonstrates that CNNs outperform traditional ANNs for image-based classification tasks and highlights the importance of regularization techniques such as dropout to improve model generalization and stability.
