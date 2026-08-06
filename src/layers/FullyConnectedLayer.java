package layers;

import math.Matrix;

public class FullyConnectedLayer implements Layer {

    private Matrix weights;
    private Matrix bias;

    public FullyConnectedLayer(int inputSize, int outputSize) {

        weights = new Matrix(inputSize, outputSize);
        weights.randomInitialize();

        bias = new Matrix(1, outputSize);
        bias.randomInitialize();
    }

    @Override
    public Matrix forward(Matrix input) {

        Matrix output = input.multiply(weights);
        output = output.add(bias);

        return output;
    }

    @Override
    public Matrix backward(Matrix gradient) {

        // Backpropagation will be implemented later
        return gradient;

    }
}