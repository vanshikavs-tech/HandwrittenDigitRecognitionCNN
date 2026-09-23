@echo off
title Handwritten Digit Recognition CNN - Concept Explainer
color 0B

:MENU
cls
echo =========================================================================================
echo       HANDWRITTEN DIGIT RECOGNITION SYSTEM - CONVOLUTIONAL NEURAL NETWORK IN JAVA        
echo                       Advanced Object Oriented Programming (AOOP)                         
echo =========================================================================================
echo.
echo   SELECT A TOPIC TO EXPLORE CONCEPTS OR LAUNCH SYSTEM ACTIONS:
echo.
echo   [1] Core Object-Oriented Programming (OOP) Principles Applied
echo   [2] Linear Algebra and The Math Engine (Matrix, Tensor, Dot Product)
echo   [3] CNN Layer Anatomy (Conv2D, ReLU, MaxPool, Flatten, Dense, Softmax)
echo   [4] Mathematics of Backpropagation, Loss, and Gradient Descent
echo   [5] MNIST Dataset Pipeline and Image Preprocessing (Center of Mass)
echo   [6] End-to-End System Architecture and Data Flow
echo.
echo   -------------------------------- SYSTEM ACTIONS ---------------------------------------
echo   [7] Run Automated Unit Test Suite (Matrix, Layers, Backprop, Toy Model)
echo   [8] Run Quick CNN Training Demo in Terminal (Main --demo)
echo   [9] Launch Interactive Java Swing GUI Drawing Canvas (Main --gui)
echo   [0] Exit
echo =========================================================================================
choice /c 1234567890 /m "Enter your choice: "

if errorlevel 10 goto EXIT
if errorlevel 9 goto RUN_GUI
if errorlevel 8 goto RUN_DEMO
if errorlevel 7 goto RUN_TESTS
if errorlevel 6 goto ARCHITECTURE
if errorlevel 5 goto PREPROCESS
if errorlevel 4 goto BACKPROP
if errorlevel 3 goto LAYERS
if errorlevel 2 goto MATH
if errorlevel 1 goto OOP
goto MENU

:OOP
cls
echo =========================================================================================
echo               TOPIC 1: OBJECT-ORIENTED PROGRAMMING (OOP) PRINCIPLES APPLIED
echo =========================================================================================
echo.
echo 1. ABSTRACTION:
echo    - 'Layer.java' defines an abstract contract:
echo         Tensor forward(Tensor input);
echo         Tensor backward(Tensor outputGradient);
echo         void updateWeights(double learningRate);
echo    - Calling layers only need to know WHAT a layer does, not HOW convolution or
echo      matrix multiplication is internally computed.
echo.
echo 2. POLYMORPHISM:
echo    - 'NeuralNetwork' maintains a polymorphic list: 'List^<Layer^> layers'.
echo    - When looping forward or backward, Java executes the subclass implementation:
echo         for (Layer l : layers) { current = l.forward(current); }
echo    - 'ConvolutionLayer', 'MaxPoolingLayer', and 'FullyConnectedLayer' all execute
echo      their specialized logic interchangeably through the same interface.
echo.
echo 3. ENCAPSULATION:
echo    - Weights, biases, and gradient accumulators are declared 'private'.
echo    - 'Matrix.java' and 'Tensor.java' encapsulate their internal arrays (double[][],
echo      double[][][]), protecting dimensional integrity via strict boundary validation.
echo.
echo 4. COMPOSITION and PIPELINE PATTERN:
echo    - 'NeuralNetwork' is composed of multiple Layers.
echo    - 'Trainer' composes a 'NeuralNetwork' and a 'LossFunction'.
echo    - 'DataSample' encapsulates an image Tensor and label target.
echo.
echo 5. OBSERVER PATTERN:
echo    - 'TrainingCallback' allows decoupled notification from the training engine
echo      to the console or graphical Swing progress bars.
echo.
echo =========================================================================================
pause
goto MENU

:MATH
cls
echo =========================================================================================
echo             TOPIC 2: LINEAR ALGEBRA and THE MATH ENGINE (MATRIX and TENSOR)
echo =========================================================================================
echo.
echo 1. MATRIX (2D Engine):
echo    - Represents 2D numerical grids (rows x cols).
echo    - Key Operations:
echo      * Matrix Multiplication: C = A x B where (M x K) * (K x N) = (M x N)
echo        Summing pairwise products: C[i][j] = sum_k(A[i][k] * B[k][j]).
echo      * Transposition: (A^T)[j][i] = A[i][j] (swaps rows and columns).
echo        Crucial for calculating backpropagation weight updates.
echo      * Hadamard Product: Element-wise multiplication A .* B.
echo      * Xavier / He Weight Initialization: Scales weights by sqrt(6/(fanIn+fanOut))
echo        to prevent vanishing or exploding activations across deep layers.
echo.
echo 2. TENSOR (3D Engine):
echo    - Represents multi-channel volumes: (channels x height x width).
echo    - Grayscale MNIST image = 1 channel x 28 height x 28 width.
echo    - Feature maps = C channels x H height x W width.
echo    - Slicing: Extracts 2D matrix channels or reshapes into flattened 1D vectors.
echo.
echo =========================================================================================
pause
goto MENU

:LAYERS
cls
echo =========================================================================================
echo               TOPIC 3: NEURAL NETWORK and CNN LAYER ANATOMY
echo =========================================================================================
echo.
echo 1. CONVOLUTIONAL LAYER ('ConvolutionLayer.java'):
echo    - Slides small learnable filters (e.g. 3x3) over input channels.
echo    - Detects spatial features: edges, angles, loops, and digit strokes.
echo    - Equation: Output[f][r][c] = Bias[f] + sum(Input .* Filter[f]).
echo.
echo 2. RECTIFIED LINEAR UNIT ('ReLULayer.java'):
echo    - Non-linear activation: f(x) = max(0, x).
echo    - Why? Prevents linear collapse of stacked layers, allowing the network to
echo      learn non-linear decision boundaries.
echo.
echo 3. MAX POOLING LAYER ('MaxPoolingLayer.java'):
echo    - Downsamples spatial dimensions (e.g. 2x2 window with stride 2).
echo    - Selects the dominant feature, cuts compute by 75%%, and adds translation tolerance.
echo.
echo 4. FLATTEN LAYER ('FlattenLayer.java'):
echo    - Converts 3D spatial tensor (C x H x W) into a 1D vector (1 x C*H*W).
echo    - Seamlessly bridges feature extraction (Conv/Pool) to classification (Dense).
echo.
echo 5. FULLY CONNECTED / DENSE LAYER ('FullyConnectedLayer.java'):
echo    - Linear transformation: Y = X * W + B.
echo    - Connects every incoming feature to every output neuron.
echo.
echo 6. SOFTMAX LAYER ('SoftmaxLayer.java'):
echo    - Converts raw output logits into a normalized probability distribution:
echo      P_i = exp(z_i - max(z)) / sum_j(exp(z_j - max(z))).
echo    - Sum of all 10 probabilities equals exactly 1.0 (100%%).
echo.
echo =========================================================================================
pause
goto MENU

:BACKPROP
cls
echo =========================================================================================
echo          TOPIC 4: MATHEMATICS OF BACKPROPAGATION, LOSS, and GRADIENT DESCENT
echo =========================================================================================
echo.
echo 1. LOSS FUNCTION ('CrossEntropyLoss.java'):
echo    - Measures divergence between predicted probability P and ground truth Y:
echo      Loss = - sum_i ( Y_i * ln(P_i) )
echo    - Combined Softmax + Cross-Entropy Loss gradient:
echo      dL/dZ = (P - Y)  [Predicted Probabilities minus One-Hot Target]
echo.
echo 2. BACKPROPAGATION (CHAIN RULE CALCULUS):
echo    - Fully Connected Layer:
echo        dL/dW = X^T * dY   (Gradients with respect to weights)
echo        dL/dB = sum(dY)    (Gradients with respect to biases)
echo        dL/dX = dY * W^T   (Gradients passed backward to previous layer)
echo    - ReLU Layer:
echo        dL/dX = dY if X > 0 else 0 (Derivative gating).
echo    - Max Pooling Layer:
echo        Routes incoming gradient exclusively to the argmax position; 0 elsewhere.
echo    - Convolution Layer:
echo        dL/dK = input * dY  (Accumulates kernel gradient)
echo        dL/dX = dY (convolved with transposed filter)
echo.
echo 3. GRADIENT DESCENT (OPTIMIZATION):
echo    - Parameters updated in opposite direction of gradient:
echo        W_new = W_old - (learning_rate * dW)
echo        B_new = B_old - (learning_rate * dB)
echo.
echo =========================================================================================
pause
goto MENU

:PREPROCESS
cls
echo =========================================================================================
echo        TOPIC 5: MNIST DATASET PIPELINE and IMAGE PREPROCESSING (CENTER OF MASS)
echo =========================================================================================
echo.
echo 1. MNIST DATASET FORMAT:
echo    - 70,000 grayscale handwritten digits (0 to 9), size 28x28 pixels.
echo    - Stored in IDX binary format (magic numbers 2051 for images, 2049 for labels).
echo    - 'MNISTLoader' parses raw byte streams and normalizes pixels from [0,255] to [0.0,1.0].
echo.
echo 2. WHY USER DRAWING PREPROCESSING IS ESSENTIAL:
echo    - If a user draws a digit small or in a corner, an uncentered CNN fails.
echo    - 'ImagePreprocessor.java' performs the exact 4-step MNIST standardization:
echo      Step 1: Bounding Box Detection - isolates non-black drawn stroke pixels.
echo      Step 2: Proportional Aspect-Ratio Scaling - scales stroke into a 20x20 box.
echo      Step 3: Center of Mass (Centroid) Calculation:
echo              X_c = sum(x * pixel_value) / sum(pixel_value)
echo              Y_c = sum(y * pixel_value) / sum(pixel_value)
echo      Step 4: Centering onto 28x28 Canvas - shifts centroid to center point (14, 14).
echo.
echo =========================================================================================
pause
goto MENU

:ARCHITECTURE
cls
echo =========================================================================================
echo                 TOPIC 6: END-TO-END SYSTEM ARCHITECTURE and DATA FLOW
echo =========================================================================================
echo.
echo [TRAINING PIPELINE]
echo   MNIST IDX / Synthetic Data
echo        ^|
echo        v
echo   28 x 28 Tensor (1 Channel)
echo        ^|
echo        v
echo   ConvLayer (4 filters, 3x3) ---^> Feature Maps (4 x 26 x 26)
echo        ^|
echo        v
echo   ReLULayer -------------------^> Non-linear Activations (4 x 26 x 26)
echo        ^|
echo        v
echo   MaxPoolingLayer (2x2) -------^> Downsampled Features (4 x 13 x 13)
echo        ^|
echo        v
echo   FlattenLayer ----------------^> 1D Vector (676 values)
echo        ^|
echo        v
echo   FullyConnectedLayer ---------^> Hidden Neurons (32 values)
echo        ^|
echo        v
echo   ReLULayer
echo        ^|
echo        v
echo   FullyConnectedLayer ---------^> Raw Logits (10 values)
echo        ^|
echo        v
echo   SoftmaxLayer ----------------^> Probabilities for [0..9]
echo        ^|
echo        +--^> CrossEntropyLoss --^> Backpropagation --^> Weight Updates
echo.
echo [INTERACTIVE INFERENCE PIPELINE]
echo   User Mouse Drawing --^> DrawingPanel (280x280)
echo        ^|
echo        v
echo   ImagePreprocessor (Crop, Scale 20x20, Center-of-Mass) --^> 28x28 Tensor
echo        ^|
echo        v
echo   Trained CNN Forward Pass --^> Digit Prediction + Confidence %% + Bar Chart
echo.
echo =========================================================================================
pause
goto MENU

:RUN_TESTS
cls
echo =========================================================================================
echo                      RUNNING AUTOMATED UNIT TEST SUITE
echo =========================================================================================
echo.
echo Compiling source and test files...
javac -d bin -sourcepath src src/Main.java test/NetworkTest.java
if errorlevel 1 (
    echo Compilation failed!
    pause
    goto MENU
)
echo Compilation successful. Running tests...
echo.
java -cp bin test.NetworkTest
echo.
pause
goto MENU

:RUN_DEMO
cls
echo =========================================================================================
echo                     RUNNING QUICK CNN TRAINING and TERMINAL DEMO
echo =========================================================================================
echo.
echo Compiling...
javac -d bin -sourcepath src src/Main.java test/NetworkTest.java
echo Running Main --demo ...
echo.
java -cp bin Main --demo
echo.
pause
goto MENU

:RUN_GUI
cls
echo =========================================================================================
echo                    LAUNCHING INTERACTIVE SWING GUI DRAWING CANVAS
echo =========================================================================================
echo.
echo Compiling...
javac -d bin -sourcepath src src/Main.java test/NetworkTest.java
echo Launching GUI...
start javaw -cp bin Main --gui
echo GUI launched in separate window!
timeout /t 3 >nul
goto MENU

:EXIT
cls
echo Thank you for exploring the Handwritten Digit Recognition CNN System.
exit /b 0
