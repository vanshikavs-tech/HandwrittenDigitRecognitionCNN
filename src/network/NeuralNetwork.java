package network;

import java.util.ArrayList;
import java.util.List;

import math.Matrix;
import layers.Layer;

public class NeuralNetwork {

    private List<Layer> layers;

    public NeuralNetwork() {
        layers = new ArrayList<>();
    }

    // Add a layer
    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    // Forward pass
    public Matrix predict(Matrix input) {

        Matrix output = input;

        for (Layer layer : layers) {
            output = layer.forward(output);
        }

        return output;
    }

    public void updateWeights(double learningRate) {

    for (Layer layer : layers) {
        layer.updateWeights(learningRate);
    }

}
}