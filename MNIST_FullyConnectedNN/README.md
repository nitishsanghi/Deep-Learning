# MNIST Neural Network for Digit Classification

## Project Overview
This project develops a neural network to classify handwritten digits from the MNIST dataset. The focus is on building a robust model capable of achieving high accuracy levels, commonly seen with more complex convolutional neural networks. This implementation serves as a benchmark for understanding model performance on a well-established dataset.

## Key Features
- **Neural Network Architecture**: Multiple layers including convolutional, dropout, and max pooling layers.
- **Dataset**: MNIST, containing images of handwritten digits.
- **Objective**: To classify digits accurately and understand the dataset's complexity.
- **Technology Stack**: PyTorch, torchvision, NumPy, Matplotlib.

## Setup Instructions

### Prerequisites
- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib

### Installation
1. Clone this repository to your local machine.
2. Ensure the required libraries are installed. You can install them via `pip`:
   ```sh
   pip install torch torchvision numpy matplotlib
   ```
3. Access the MNIST dataset through PyTorch's `torchvision.datasets`, ensuring you set `download=True` in your dataset loader.

### Execution
Execute the Jupyter Notebook `MNISTModelTraining.ipynb` using Jupyter Notebook or Google Colab, running cells sequentially to ensure proper setup and execution flow.

## Model Architecture
The model consists of sequential layers designed to progressively reduce the spatial size while increasing the depth of feature maps:

### Convolutional Layers
- **Input**: 28x28 grayscale images.
- **Processing**: Two convolutional layers with ReLU activation, followed by max pooling and dropout layers to reduce overfitting and dimensionality.

### Fully Connected Layers
- **Details**: Layers to classify the features extracted by the convolutional layers into one of the 10 digit classes.
- **Activation**: The final output layer uses the softmax function to provide probabilities for each class.

## Training Process
- **Loss Function**: Cross-Entropy Loss, suitable for classification tasks.
- **Optimizer**: Adam, chosen for efficient computation and adaptive learning rate capabilities.
- **Epochs**: Trained over 10 epochs with a batch size of 96.

## Results

### Classification Accuracy
Post-training, the model achieves an accuracy that highlights its capability to effectively classify the digits from the MNIST dataset. The accuracy and loss metrics are plotted to demonstrate the model's learning progression over epochs.

## Discussion
This project validates the neural network's design by achieving high classification accuracy on the MNIST dataset. Adjustments in the network architecture or training parameters could be explored to further enhance performance.

## Conclusion
The implementation showcases the effectiveness of convolutional neural networks in image classification tasks. The high accuracy on the MNIST dataset reaffirms the model's capability, making it a valuable reference for further experiments and educational purposes.
