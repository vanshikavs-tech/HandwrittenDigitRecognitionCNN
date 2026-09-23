package network;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import layers.Layer;
import layers.SoftmaxLayer;
import math.Matrix;
import math.Tensor;

/**
 * NeuralNetwork orchestrates a sequential chain of Layer components.
 * Manages the complete lifecycle:
 *   1. Forward pass (inference)
 *   2. Backward pass (backpropagation)
 *   3. Parameter update (gradient descent)
 * Demonstrates the Composite and Pipeline design patterns in OOP.
 */
public class NeuralNetwork implements Serializable {
    private static final long serialVersionUID = 1L;

    private final List<Layer> layers;

    public NeuralNetwork() {
        this.layers = new ArrayList<>();
    }

    public void addLayer(Layer layer) {
        if (layer == null) {
            throw new IllegalArgumentException("Cannot add a null layer.");
        }
        layers.add(layer);
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public Layer getLayer(int index) {
        return layers.get(index);
    }

    public int getLayerCount() {
        return layers.size();
    }

    /**
     * Executes the forward pass across all layers sequentially.
     */
    public Tensor forward(Tensor input) {
        Tensor current = input;
        for (Layer layer : layers) {
            current = layer.forward(current);
        }
        return current;
    }

    /**
     * Executes backpropagation in reverse layer order.
     * If the final layer is Softmax and the gradient is already dL/dZ = (P - Y),
     * backpropagation seamlessly routes dZ directly to the penultimate layer.
     */
    public Tensor backward(Tensor lossGradient) {
        Tensor currentGrad = lossGradient;
        int lastIndex = layers.size() - 1;

        // If the last layer is Softmax and loss gradient is already (P - Y)
        if (lastIndex >= 0 && layers.get(lastIndex) instanceof SoftmaxLayer) {
            lastIndex--;
        }

        for (int i = lastIndex; i >= 0; i--) {
            currentGrad = layers.get(i).backward(currentGrad);
        }
        return currentGrad;
    }

    /**
     * Updates weights across all parameterized layers using gradient descent.
     */
    public void updateWeights(double learningRate) {
        for (Layer layer : layers) {
            if (layer.hasWeights()) {
                layer.updateWeights(learningRate);
            }
        }
    }

    /**
     * Predicts probabilities for a Tensor input.
     */
    public Tensor predict(Tensor input) {
        return forward(input);
    }

    /**
     * Compatibility overload for 2D Matrix input.
     */
    public Matrix predict(Matrix input) {
        Tensor inTensor = Tensor.fromMatrix(input);
        Tensor outTensor = predict(inTensor);
        return outTensor.toMatrix();
    }

    /**
     * Returns the predicted class label (index of maximum probability).
     */
    public int predictDigit(Tensor input) {
        Tensor output = predict(input);
        return output.argMax();
    }

    /**
     * Returns the probability distribution as a double array.
     */
    public double[] predictProbabilities(Tensor input) {
        Tensor output = predict(input);
        return output.toFlatArray();
    }

    /**
     * Returns the confidence score (highest probability) for the predicted digit.
     */
    public double predictConfidence(Tensor input) {
        double[] probs = predictProbabilities(input);
        double maxProb = 0.0;
        for (double p : probs) {
            if (p > maxProb) {
                maxProb = p;
            }
        }
        return maxProb;
    }

    /**
     * Displays a clean summary of the network architecture.
     */
    public void printSummary() {
        System.out.println("==========================================================");
        System.out.println("                NEURAL NETWORK ARCHITECTURE               ");
        System.out.println("==========================================================");
        for (int i = 0; i < layers.size(); i++) {
            Layer l = layers.get(i);
            System.out.printf("[%2d] %-25s Trainable: %s%n",
                i + 1, l.getClass().getSimpleName(), l.hasWeights() ? "YES" : "NO");
        }
        System.out.println("==========================================================");
    }
}