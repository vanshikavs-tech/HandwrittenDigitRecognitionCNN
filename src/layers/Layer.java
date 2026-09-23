package layers;

import java.io.Serializable;
import math.Tensor;

/**
 * Common interface for all neural network layers.
 * Enforces the contract for the Forward Pass, Backward Pass (Backpropagation),
 * and Weight Updates (Gradient Descent), embodying the OOP principles of
 * Abstraction and Polymorphism.
 */
public interface Layer extends Serializable {

    /**
     * Executes the forward pass: computes layer output given an input tensor.
     * Caches intermediate activations required for backpropagation.
     *
     * @param input the input Tensor
     * @return the transformed output Tensor
     */
    Tensor forward(Tensor input);

    /**
     * Executes the backward pass (backpropagation): computes the gradient of the loss
     * with respect to the layer's inputs, and accumulates gradients for learnable parameters.
     *
     * @param outputGradient gradient of loss with respect to this layer's output (dL/dY)
     * @return gradient of loss with respect to this layer's input (dL/dX)
     */
    Tensor backward(Tensor outputGradient);

    /**
     * Updates the learnable parameters (weights and biases) using accumulated gradients
     * according to the learning rate.
     *
     * @param learningRate step size for gradient descent
     */
    default void updateWeights(double learningRate) {}

    /**
     * Indicates whether this layer has trainable parameters.
     *
     * @return true if layer contains weights/biases, false otherwise
     */
    default boolean hasWeights() {
        return false;
    }
}