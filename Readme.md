# Handwritten Digit Recognition CNN (Pure Java)

An end-to-end **Convolutional Neural Network (CNN)** and interactive handwritten digit recognition system built **entirely from scratch in pure Java SE (zero external machine learning libraries)**.

Developed as an **Advanced Object-Oriented Programming (AOOP)** final capstone project.

---

## Authors & Contributors
- **Avani Hariya** (RA2511056010006)
- **Stuti Rai** (RA2511056010012)
- **Vanshika Singh** (RA2511056010003)

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
- **Interactive Concept Explainer**: Run `CONCEPTS.cmd` to explore interactive theory, calculus, and architecture walkthroughs directly in Windows Command Prompt.

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
│   │   └── MNISTLoader.java         // Binary IDX reader, downloader & generator
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
│   └── digit-cnn.model              // Serialized trained model
│
├── CONCEPTS.cmd                     // Interactive Windows CMD concept explainer
├── CONCEPTS.md                      // Comprehensive mathematical & OOP guide
└── README.md
```

---

## Quick Start & Usage

### 1. Interactive Windows CMD Explainer
Double-click `CONCEPTS.cmd` or run in terminal:
```cmd
CONCEPTS.cmd
```
Provides an interactive menu explaining:
- Core OOP Principles
- Linear algebra & Math Engine
- CNN Layer Anatomy
- Backpropagation Calculus
- MNIST Preprocessing
- Quick Launchers for Tests, Demo, and GUI

### 2. Compile the Project
```cmd
javac -d bin -sourcepath src src/Main.java test/NetworkTest.java
```

### 3. Run Automated Unit Test Suite
Runs 47 automated tests covering Matrix math, Tensors, forward/backward layers, and a toy pattern learning convergence problem:
```cmd
java -cp bin test.NetworkTest
```

### 4. Run CLI Quick Demo
Trains a miniature CNN on synthetic digits in seconds and prints ASCII digit visualizations with predictions:
```cmd
java -cp bin Main --demo
```

### 5. Launch Interactive Swing GUI
```cmd
java -cp bin Main --gui
```
- Draw a digit with your mouse on the canvas.
- See the real-time preprocessed 28x28 preview.
- View the predicted digit and confidence score.
- Watch the 10-class probability distribution bar chart update live!
- Use **"Train Quick Demo"** or **"Train on MNIST"** directly from the GUI toolbar.

---

## Detailed Concepts & Theory
For full mathematical derivations, backpropagation chain rule formulas, He/Xavier initializations, and OOP design patterns, see:
👉 [**CONCEPTS.md**](file:///c:/Users/vansh/OneDrive/Desktop/HandwrittenDigitCNN/CONCEPTS.md)
