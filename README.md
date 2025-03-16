# Convolutional Neural Network from Scratch using PyTorch

## Objective
This project implements a Convolutional Neural Network (CNN) from scratch using the PyTorch framework to classify handwritten digits. The dataset consists of 60,000 images for training and 10,000 images for testing. The implementation follows the given network architecture and training parameters.

## Dataset
The MNIST dataset consists of grayscale images of handwritten digits (0-9). It is divided into a training set (60,000 images) and a test set (10,000 images). The dataset is publicly available and can be accessed online.

## Network Architecture

| Layer                 | Kernel Size | Pooling  | Stride | Output Channels | Activation |
|----------------------|------------|---------|--------|----------------|------------|
| Convolutional Layer 1 | 7x7        | Max     | 1      | 16             | ReLU       |
| Convolutional Layer 2 | 5x5        | Max     | 1      | 8              | ReLU       |
| Convolutional Layer 3 | 3x3        | Average | 2      | 4              | ReLU       |
| Fully Connected Layer | -          | -       | -      | 10             | Softmax    |


## Training Details

- **Padding:** Zero padding is used to preserve input dimensions.  
- **Train-test split:** Standard dataset split is used.  
- **Batch size:  32**  
- **Activation function:** ReLU for convolutional layers, Softmax for output layer.  
- **Optimizer:** Adam optimizer.  
- **Loss function:** Cross-entropy loss.  
- **Number of epochs:** 10 (can be increased if accuracy is low).  
- **Evaluation metrics:** Accuracy, loss per epoch, confusion matrix.  
- **Total trainable and non-trainable parameters:** Reported at the end of training.  


## Experiment 1: 10-Class Classification

In this experiment, the CNN model is trained and evaluated on the original dataset containing 10 classes (digits 0-9). The performance is analyzed using accuracy plots, loss plots, and a confusion matrix.

## Experiment 2: 4-Class Classification

For this experiment, the original 10 classes are combined into 4 broader classes:

- **Class 1:** {0, 6}

- **Class 2:** {1, 7}

- **Class 3:** {2, 3, 8, 5}

- **Class 4:** {4, 9}

The CNN model from Experiment 1 is used to perform this new classification task. Performance is analyzed similarly to Experiment 1.

## Results & Evaluation

Accuracy and loss graphs: Plotted per epoch for both experiments.

Confusion matrix: Generated for both 10-class and 4-class classification tasks.

Parameter report: Total trainable and non-trainable parameters are reported.

Observations: Analysis of results, error analysis, and recommendations for improvement.






