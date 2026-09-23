package utils;

import layers.ConvolutionLayer;
import layers.FlattenLayer;
import layers.FullyConnectedLayer;
import layers.MaxPoolingLayer;
import layers.ReLULayer;
import layers.SoftmaxLayer;
import network.NeuralNetwork;

/**
 * ModelBuilder provides factory methods for standard neural network architectures:
 * Multilayer Perceptrons (MLP) and Convolutional Neural Networks (CNN).
 * Demonstrates the Factory and Builder design patterns.
 */
public class ModelBuilder {

    /**
     * Constructs a fast, effective Convolutional Neural Network for MNIST digit recognition.
     * Architecture:
     *   Input: 1 x 28 x 28
     *   1. Conv (4 filters, 3x3 kernel, stride 1, pad 0) -> 4 x 26 x 26
     *   2. ReLU
     *   3. MaxPool (2x2, stride 2)                        -> 4 x 13 x 13
     *   4. Conv (8 filters, 3x3 kernel, stride 1, pad 0) -> 8 x 11 x 11
     *   5. ReLU
     *   6. MaxPool (2x2, stride 2)                        -> 8 x 5 x 5 (200 features)
     *   7. Flatten                                        -> 200 vector
     *   8. Dense (200 -> 32)
     *   9. ReLU
     *   10. Dense (32 -> 10)
     *   11. Softmax                                       -> 10 class probabilities
     */
    public static NeuralNetwork buildCNN() {
        NeuralNetwork net = new NeuralNetwork();

        net.addLayer(new ConvolutionLayer(1, 4, 3, 1, 0));
        net.addLayer(new ReLULayer());
        net.addLayer(new MaxPoolingLayer(2, 2));

        net.addLayer(new ConvolutionLayer(4, 8, 3, 1, 0));
        net.addLayer(new ReLULayer());
        net.addLayer(new MaxPoolingLayer(2, 2));

        net.addLayer(new FlattenLayer());
        net.addLayer(new FullyConnectedLayer(8 * 5 * 5, 32));
        net.addLayer(new ReLULayer());
        net.addLayer(new FullyConnectedLayer(32, 10));
        net.addLayer(new SoftmaxLayer());

        return net;
    }

    /**
     * Constructs a compact, fast single-conv-stage CNN for rapid training.
     */
    public static NeuralNetwork buildFastCNN() {
        NeuralNetwork net = new NeuralNetwork();

        // 1x28x28 -> 4x13x13 (pool)
        net.addLayer(new ConvolutionLayer(1, 4, 3, 1, 0)); // 4x26x26
        net.addLayer(new ReLULayer());
        net.addLayer(new MaxPoolingLayer(2, 2));            // 4x13x13 = 676
        net.addLayer(new FlattenLayer());
        net.addLayer(new FullyConnectedLayer(4 * 13 * 13, 32));
        net.addLayer(new ReLULayer());
        net.addLayer(new FullyConnectedLayer(32, 10));
        net.addLayer(new SoftmaxLayer());

        return net;
    }

    /**
     * Constructs a high-performance Multilayer Perceptron (MLP).
     * Architecture: 784 -> 128 -> 64 -> 10.
     */
    public static NeuralNetwork buildMLP() {
        NeuralNetwork net = new NeuralNetwork();

        net.addLayer(new FlattenLayer());
        net.addLayer(new FullyConnectedLayer(784, 128));
        net.addLayer(new ReLULayer());
        net.addLayer(new FullyConnectedLayer(128, 64));
        net.addLayer(new ReLULayer());
        net.addLayer(new FullyConnectedLayer(64, 10));
        net.addLayer(new SoftmaxLayer());

        return net;
    }
}
