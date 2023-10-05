# MNIST Autoencoder for Digit Reconstruction and Anomaly Detection

## Project Overview
This project builds a linear autoencoder to compress and reconstruct digits from the MNIST dataset. Through this process, we explore the capabilities of autoencoders in learning efficient representations and reconstructing input data with minimal loss. Additionally, we demonstrate the use of the trained model for anomaly detection, identifying digits that significantly differ from the norm based on reconstruction loss.

## Key Features
- **Autoencoder Architecture**: Linear layers for encoding and decoding, with ReLU and Batch Normalization.
- **Dataset**: MNIST, a collection of handwritten digits.
- **Objective**: To learn compressed representations of digits and to identify anomalies.
- **Technology Stack**: PyTorch, NumPy, Pandas, TQDM.

## Setup Instructions

### Prerequisites
- Python 3.6 or later
- PyTorch
- NumPy
- Pandas
- TQDM

### Installation
1. Clone this repository to your local machine.
2. Ensure you have the required libraries installed. You can install them using `pip`:
   ```sh
   pip install torch numpy pandas tqdm
   ```
3. Download the MNIST dataset or access it through PyTorch's `torchvision.datasets`.

### Execution
Run the notebook `LinearEncoderDecoderMNIST.ipynb` using Jupyter Notebook or Google Colab. Ensure all cells are executed in order.

## Model Architecture
The autoencoder is composed of an encoder and a decoder module, each consisting of linear layers, ReLU activation functions, and batch normalization.

### Encoder
- Input: Flattened 28x28 MNIST images (784 features).
- Layers: Three linear layers with dimensions reducing from 784 to 256, then to 128, and finally to a specified encoding dimension.
- Regularization: Batch normalization is applied after the first two ReLU activations to stabilize learning.

### Decoder
- Input: Encoded representations from the encoder.
- Layers: Mirrors the encoder structure in reverse, gradually expanding the encoding back to the original dimensionality.
- Activation: The final layer uses a Sigmoid activation to ensure the output values are in the [0, 1] range, matching the input image's normalization.

### Forward Pass
- The input images are flattened, passed through the encoder to get reduced representations, then through the decoder to reconstruct the images.

## Training Process
- **Loss Function**: Mean Squared Error (MSE) for comparing the original and reconstructed images.
- **Optimizer**: Adam, with a learning rate of 0.05.
- **Epochs**: The model was trained for 25 epochs with a batch size of 1024.

![Training and Validation Loss](https://github.com/nitishsanghi/Deep-Learning-Autoencoders/blob/main/MNIST/LinearAutoencoderTrainingLoss.png)

*Figure 1: Training and Validation Loss over Epochs*

## Results

### Digit Reconstruction
The autoencoder was capable of reconstructing the MNIST digits with high fidelity. The comparison between original and reconstructed images shows the model's effectiveness.

![Digit Reconstruction](https://github.com/nitishsanghi/Deep-Learning-Autoencoders/blob/main/MNIST/LinearAutoencoderEasyToReconstruct.png)

*Figure 2: Original vs. Reconstructed Digits*

### Anomaly Detection
By examining reconstruction loss, the model could identify digits that deviated significantly from the typical patterns it learned during training.

![Anomaly Detection](https://github.com/nitishsanghi/Deep-Learning-Autoencoders/blob/main/MNIST/LinearAutoencoderDifficultToReconstruct.png)

*Figure 3: Anomaly Detection based on Reconstruction Loss*

## Discussion
The project demonstrates the utility of linear autoencoders in data compression and anomaly detection. While the model achieves high accuracy in reconstructing digits, the simplicity of the architecture limits its capacity. Future work could explore more complex models, including convolutional autoencoders, to capture spatial hierarchies in images for enhanced performance.

## Conclusion
This exploration into linear autoencoders with the MNIST dataset highlights the balance between data compression and reconstruction fidelity. The added application of anomaly detection showcases the model's versatility and opens avenues for further research and development in unsupervised learning tasks.
