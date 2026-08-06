package layers;

import math.Matrix;

public interface Layer {

    // Forward pass
    Matrix forward(Matrix input);

    // Backward pass
    Matrix backward(Matrix gradient);

    // Update weights (does nothing for layers without weights)
    void updateWeights(double learningRate);

}