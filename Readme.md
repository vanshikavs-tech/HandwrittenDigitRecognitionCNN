# Handwritten Digit Recognition CNN (Pure Java)

An end-to-end **Convolutional Neural Network (CNN)** and interactive handwritten digit recognition system built **entirely from scratch in pure Java SE (zero external machine learning libraries)**.

Developed as an **Advanced Object-Oriented Programming (AOOP)** final capstone project.

---

## Authors & Contributors
- **Avani Hariya** (RA2511056010006)
- **Vanshika Singh** (RA2511056010003)
- **Stuti Rai** (RA2511056010012)

---

## Key Features

- **Pure Java Implementation**: Zero PyTorch, TensorFlow, Weka, or DL4J dependencies. Built solely with standard Java SE libraries (`java.awt`, `javax.swing`, `java.io`, `java.util`).
- **Complete Math Engine**: 2D `Matrix` and 3D `Tensor` engines supporting dot products, transpositions, Hadamard multiplication, channel slicing, and Xavier/He normal weight initializations.
- **Full Neural Network Layers**:
  - `ConvolutionLayer` (multi-channel 2D convolution, padding, stride, analytical backprop)
  - `MaxPoolingLayer` (2x2 downsampling with argmax gradient routing)
  - `FlattenLayer` (3D Tensor to 1D/2D vector reshaping and exact gradient restoration)
  - `FullyConnectedLayer` (Dense layer with $dW = X^T \cdot dY$, $dB = \sum dY$, $dX = dY \cdot W^T$)
  - `ReLULayer` (Rectified Linear Unit activation with derivative gating)
  - `SoftmaxLayer` (Numerically stable softmax with analytical Jacobian backward propagation)
- **Training Engine**: Mini-batch Stochastic Gradient Descent (SGD) with categorical Cross-Entropy loss.
- **MNIST Loader & Downloader**: Parses binary IDX format directly, automatically downloads and extracts gzip files if missing, and includes an offline synthetic digit dataset generator.
- **Image Preprocessor**: Real-time bounding-box extraction, aspect-ratio scaling to 20x20, and **center-of-mass centroid alignment** onto a 28x28 canvas (exact MNIST standard).
- **Interactive Swing GUI**:
  - High-precision drawing canvas with antialiased brush strokes
  - Real-time preprocessed 28x28 tensor preview
  - Live digit prediction with confidence score (%)
  - 10-class probability distribution bar chart
  - Model serialization (Save & Load models)

---

## Directory Structure

```text
HandwrittenDigitCNN/
│
├── src/
│   ├── math/
│   │   ├── Matrix.java              // 2D linear algebra engine
│   │   └── Tensor.java              // 3D volumetric tensor engine
│   │
│   ├── layers/
│   │   ├── Layer.java               // Unified OOP Layer interface
│   │   ├── FullyConnectedLayer.java // Dense layer with backpropagation
│   │   ├── ReLULayer.java           // ReLU activation with derivative gating
│   │   ├── SoftmaxLayer.java        // Softmax with numerical stability trick
│   │   ├── ConvolutionLayer.java    // Multi-filter 2D/3D convolution & backprop
│   │   ├── MaxPoolingLayer.java     // 2x2 max pooling with argmax routing
│   │   ├── FlattenLayer.java        // Tensor to Matrix reshaping
│   │   └── IdentityLayer.java       // Passthrough layer for testing
│   │
│   ├── network/
│   │   └── NeuralNetwork.java       // Sequential pipeline orchestrator
│   │
│   ├── loss/
│   │   ├── LossFunction.java        // Loss interface
│   │   └── CrossEntropyLoss.java    // Categorical cross-entropy & logit gradient
│   │
│   ├── trainer/
│   │   ├── Trainer.java             // Mini-batch SGD training loop
│   │   └── TrainingCallback.java    // Progress listener for console & GUI
│   │
│   ├── dataset/
│   │   ├── DataSample.java          // Sample wrapper (Tensor image + label)
│   │   ├── MNISTLoader.java         // Binary IDX reader, downloader & generator
│   │   └── DevanagariDatasetLoader.java // Devanagari digit dataset loader
│   │
│   ├── preprocessing/
│   │   └── ImagePreprocessor.java   // Bounding box + center-of-mass centering
│   │
│   ├── model/
│   │   └── ModelSerializer.java     // Model persistence (save/load weights)
│   │
│   ├── gui/
│   │   ├── DrawingPanel.java        // Antialiased interactive canvas
│   │   ├── DigitVisualizer.java     // ASCII terminal digit visualizer
│   │   └── MainWindow.java          // Main Swing GUI application
│   │
│   ├── utils/
│   │   └── ModelBuilder.java        // CNN and MLP architecture factories
│   │
│   └── Main.java                    // Main entry point
│
├── test/
│   └── NetworkTest.java             // Automated unit test suite (47 tests)
│
├── models/
│   ├── digit-cnn.model              // Serialized MNIST-trained CNN model
│   └── devanagari-digit-cnn.model   // Serialized Devanagari-trained CNN model
│
├── data/
│   ├── mnist/                       // MNIST IDX binary files (auto-downloaded)
│   └── devanagari-csv/              // Devanagari CSV dataset
│
├── scripts/
│   └── download_devanagari_dataset.sh  // Devanagari dataset download script
│
└── README.md
```

---

## Quick Start & Usage

### 1. Compile the Project
```bash
javac -d bin -sourcepath src src/Main.java test/NetworkTest.java
```
> On Windows, use the same command in Command Prompt or PowerShell.

### 2. Run Automated Unit Test Suite
Runs 47 automated tests covering Matrix math, Tensors, forward/backward layers, and a toy pattern learning convergence problem:
```bash
java -cp bin test.NetworkTest
```
Or use the Main entry point:
```bash
java -cp bin Main --test
```

### 3. Run CLI Quick Demo (Synthetic Digits)
Trains a miniature CNN on synthetic digits in seconds and prints ASCII digit visualizations with predictions:
```bash
java -cp bin Main --demo
```
> The synthetic demo is for demonstrating the pipeline, not for measuring real-world handwritten-digit accuracy.

### 4. Train on MNIST (Real Handwritten Digits)
Downloads MNIST automatically if not present, trains on real handwritten digits, evaluates on test set, and saves the model:
```bash
java -cp bin Main --train 5000 5
```
- First argument: number of training samples (max 60000)
- Second argument: number of epochs
- Use `--train 2000 5` for a smaller/faster run

### 5. Train on Devanagari Digits
Trains on Devanagari script digits (requires dataset in `data/devanagari-csv/`):
```bash
java -cp bin Main --train-devanagari 5
```
- Argument: number of epochs (default 5)
- Run `scripts/download_devanagari_dataset.sh` first to fetch the dataset

### 6. Launch Interactive Swing GUI
```bash
java -cp bin Main --gui
```
- Draw a digit with your mouse on the canvas.
- See the real-time preprocessed 28x28 preview.
- View the predicted digit and **top-class probability** (%).
- Watch the 10-class probability distribution bar chart update live!
- Use **"Train Quick Demo"** or **"Train on MNIST"** directly from the GUI toolbar.
- **Important:** Train on MNIST first (`--train`) so a real model is loaded. The GUI loads `models/digit-cnn.model` by default.
