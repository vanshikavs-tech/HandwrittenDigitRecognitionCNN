package layers;

import math.Matrix;

public class IdentityLayer implements Layer {

    @Override
    public Matrix forward(Matrix input) {
        return input;
    }

    @Override
    public Matrix backward(Matrix gradient) {
        return gradient;
    }

    @Override
    public void updateWeights(double learningRate) {
        // No weights to update
    }
}