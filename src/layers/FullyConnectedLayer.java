package layers;

import math.Matrix;

public class FullyConnectedLayer implements Layer {

    private Matrix weights;
    private Matrix bias;

    // Store values from forward pass
    private Matrix input;
    private Matrix output;

    public FullyConnectedLayer(int inputSize, int outputSize) {

        weights = new Matrix(inputSize, outputSize);
        weights.randomInitialize();

        bias = new Matrix(1, outputSize);
        bias.randomInitialize();
    }

    @Override
    public Matrix forward(Matrix input) {

        // Save input for backpropagation
        this.input = input;

        output = input.multiply(weights);
        output = output.add(bias);

        return output;
    }

    @Override
    public Matrix backward(Matrix gradient) {

        // Backpropagation implementation will come next
        return gradient;
    }

    @Override
    public void updateWeights(double learningRate) {

        // Weight update implementation will come next

    }

    // Getter methods (useful for debugging)
    public Matrix getWeights() {
        return weights;
    }

    public Matrix getBias() {
        return bias;
    }

    public Matrix getInput() {
        return input;
    }

    public Matrix getOutput() {
        return output;
    }
}