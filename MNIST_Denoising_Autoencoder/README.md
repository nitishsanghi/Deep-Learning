# Denoising Autoencoder for MNIST

## Project Overview
This project develops a denoising autoencoder to clean noisy images from the MNIST dataset. The focus is on demonstrating the autoencoder's capability to reconstruct images after being corrupted by noise, providing insights into the robustness of neural networks in handling imperfect data.

## Key Features
- **Neural Network Architecture**: Includes convolutional, transpose convolutional, ReLU activation, batch normalization, and sigmoid layers.
- **Dataset**: MNIST, featuring images of handwritten digits.
- **Objective**: To reconstruct clean images from noisy inputs and evaluate the model's performance based on reconstruction loss.
- **Technology Stack**: PyTorch, NumPy, Pandas, TQDM, Matplotlib.

## Setup Instructions

### Prerequisites
- Python 3.x
- PyTorch
- NumPy
- Pandas
- TQDM
- Matplotlib

### Installation
1. Clone this repository to your local machine.
2. Install the required libraries using `pip`:
   ```sh
   pip install torch numpy pandas tqdm matplotlib
   ```
3. Utilize the MNIST dataset through PyTorch's `torchvision.datasets`, with `download=True` in your data loader setup.

### Execution
Execute the notebook `DenoisingAutoencoderMNIST.ipynb` in Jupyter Notebook or Google Colab, ensuring each cell is run in sequence for proper initialization and execution of the model.

## Model Architecture
The autoencoder is constructed with two main components:
- **Encoder**: Compresses the input image using convolutional layers followed by max pooling to reduce dimensionality.
- **Decoder**: Reconstructs the image using transpose convolutional layers that mirror the encoder's structure.

## Training Process
- **Loss Function**: Mean Squared Error (MSE), measuring the difference between the original images and the reconstructed outputs.
- **Optimizer**: Adam, known for its efficient computation and adaptability.
- **Epochs**: 20 epochs with noise addition to simulate real-world data corruption.

## Results

### Image Reconstruction
The effectiveness of the autoencoder is demonstrated through the reconstruction of noisy images, showing significant noise reduction and clarity in the restored images.

![Reconstruction](https://github.com/nitishsanghi/Deep-Learning/blob/main/MNIST_Denoising_Autoencoder/Denoised.png)

### Model Evaluation
Model performance is evaluated using training and validation loss metrics, which are plotted to show the model's learning progress.

## Discussion
The project highlights the effectiveness of denoising autoencoders in cleaning corrupted images, which can be essential for applications in digital image restoration and medical imaging.

## Conclusion
The implementation underscores the robustness of autoencoders in handling and reconstructing noisy data, providing a solid foundation for further exploration into more complex autoencoder architectures and other types of data corruption.
