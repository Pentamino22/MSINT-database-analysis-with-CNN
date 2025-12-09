# MSINT / MNIST Database Analysis with a Convolutional Neural Network (CNN)

This project implements a simple Convolutional Neural Network (CNN) in Java to analyze and classify handwritten digits from the MNIST dataset. It includes custom loaders for MNIST `.idx` files, a basic CNN implementation, and a simple Java Swing GUI for visualization. The project was developed as part of a study of machine learning algorithms and neural network architectures.

## Project Features

- Loading and parsing MNIST `.idx` image and label files  
- Implementation of a basic Convolutional Neural Network in Java  
- Forward pass, convolution operations, pooling, activation functions  
- GUI for visualizing digit predictions  
- Dataset inspection and analysis tools  
- Fully standalone Java solution without external ML frameworks  

## Project Structure
```
MSINT-database-analysis-with-CNN/
│
├── .settings/ # Eclipse project settings
├── bin/ # Compiled Java classes
├── src/ # Source code
│ ├── MNISTLoader.java # Loader for MNIST idx files
│ ├── NeuralNetworkGUI.java # Simple GUI for visualization
│ └── SimpleCNN.java # CNN implementation
│
├── .classpath # Eclipse configuration
├── .project # Eclipse project metadata
│
├── Romanov_WF_KI_MTSB-31.pdf # Related documentation
├── data.txt # Additional project-specific data
│
├── t10k-images.idx3-ubyte # MNIST test images
├── t10k-labels.idx1-ubyte # MNIST test labels
├── train-images.idx3-ubyte # MNIST training images
└── train-labels.idx1-ubyte # MNIST training labels
```
## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Pentamino22/MSINT-database-analysis-with-CNN.git
cd MSINT-database-analysis-with-CNN
```
### 2. Compile the project
```bash
javac -d bin src/*.java
```
### 3. Run the GUI
```bash
java -cp bin NeuralNetworkGUI
```


## Dataset

The project uses the standard MNIST handwritten digit dataset in its raw .idx format:

train-images.idx3-ubyte

train-labels.idx1-ubyte

t10k-images.idx3-ubyte

t10k-labels.idx1-ubyte

The MNISTLoader.java class is responsible for binary parsing of these files.

## Model Overview

This project implements a fully custom neural network in Java, designed for processing the MNIST handwritten digit dataset.  
Despite the class name *SimpleCNN*, the model is not a classical convolutional neural network — it is a **three-layer fully connected feedforward neural network** trained using mini-batch gradient descent with manually implemented backpropagation.

### Architecture

The neural network consists of the following layers:

- **Input Layer:** 784 neurons (flattened 28×28 pixel image)  
- **Hidden Layer 1:** 16 neurons (ReLU or Sigmoid activation)  
- **Hidden Layer 2:** 16 neurons (same activation as above)  
- **Output Layer:** 10 neurons (Softmax activation for classification into digits 0–9)

### Training Details

- **Training algorithm:** Stochastic Gradient Descent with mini-batches  
- **Batch size:** 50  
- **Learning rate:** 0.025  
- **Epochs:** 50  
- **Loss function:** Cross-Entropy Loss  
- **Activations:** ReLU (default) or Sigmoid, controlled through a constant flag in the code  
- **Implementation:**  
  All operations — matrix multiplication, activations, forward pass, backward pass, and weight updates — are implemented manually without using external machine learning libraries.

### Model Saving and Loading

After training, the network’s weights and biases are exported to a human-readable file (`data.txt`).  
The GUI application (`NeuralNetworkGUI`) provides the ability to:

- Load the stored model  
- Select a random sample from the MNIST test set  
- Display the digit image  
- Run a forward pass and show the model’s prediction  

This makes the project both interactive and educational.

### Visualization (GUI)

The graphical interface displays:

- A scaled MNIST digit (280×280 pixels for clarity)  
- The predicted digit and the actual label  
- A button to test a random MNIST example  

This provides an intuitive demonstration of how the trained neural network performs on real handwritten digit images.

## License

This project is licensed under the MIT License.

## Author
Pentamino22 — created for educational and research purposes.
