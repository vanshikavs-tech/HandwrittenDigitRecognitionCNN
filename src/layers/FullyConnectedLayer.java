package layers;

import math.Matrix;
import math.Tensor;

/**
 * FullyConnectedLayer (Dense Layer).
 * Computes Y = X * W + B.
 * Implements full backpropagation:
 *   dW = X^T * dY
 *   dB = sum(dY)
 *   dX = dY * W^T
 */
public class FullyConnectedLayer implements Layer {
    private static final long serialVersionUID = 1L;

    private final int inputSize;
    private final int outputSize;

    private Matrix weights;      // inputSize x outputSize
    private Matrix bias;         // 1 x outputSize

    // Gradient accumulators
    private Matrix dW;           // inputSize x outputSize
    private Matrix dB;           // 1 x outputSize
    private int gradientCount;

    // Cache from forward pass
    private transient Matrix lastInput; // 1 x inputSize

    public FullyConnectedLayer(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;

        this.weights = new Matrix(inputSize, outputSize);
        // Xavier/Glorot initialization prevents vanishing/exploding gradients
        this.weights.xavierInitialize(inputSize, outputSize);

        this.bias = Matrix.zeros(1, outputSize);

        this.dW = Matrix.zeros(inputSize, outputSize);
        this.dB = Matrix.zeros(1, outputSize);
        this.gradientCount = 0;
    }

    @Override
    public Tensor forward(Tensor input) {
        // Flatten input tensor if needed to a 1 x inputSize Matrix
        Matrix inMatrix = (input.getChannels() == 1 && input.getHeight() == 1)
                ? input.toMatrix()
                : input.flattenToMatrix();

        Matrix outMatrix = forward(inMatrix);
        return Tensor.fromMatrix(outMatrix);
    }

    /**
     * Matrix-based forward pass: Y = X * W + B
     */
    public Matrix forward(Matrix input) {
        if (input.getCols() != inputSize) {
            throw new IllegalArgumentException(
                "Input dimension mismatch in FullyConnectedLayer. Expected: " +
                inputSize + ", got: " + input.getCols()
            );
        }
        // Cache input for backpropagation
        this.lastInput = input.copy();

        // Linear transformation: Y = X * W + B
        Matrix output = input.multiply(weights);
        output = output.add(bias);
        return output;
    }

    @Override
    public Tensor backward(Tensor outputGradient) {
        Matrix gradMatrix = (outputGradient.getChannels() == 1 && outputGradient.getHeight() == 1)
                ? outputGradient.toMatrix()
                : outputGradient.flattenToMatrix();

        Matrix inGradMatrix = backward(gradMatrix);
        return Tensor.fromMatrix(inGradMatrix);
    }

    /**
     * Matrix-based backward pass:
     *   dW = X^T * dY
     *   dB = dY
     *   dX = dY * W^T
     */
    public Matrix backward(Matrix outputGradient) {
        if (lastInput == null) {
            throw new IllegalStateException("Cannot run backward before forward pass.");
        }
        if (outputGradient.getCols() != outputSize) {
            throw new IllegalArgumentException(
                "Output gradient dimension mismatch. Expected: " +
                outputSize + ", got: " + outputGradient.getCols()
            );
        }

        // dW = X^T * dY (inputSize x 1) * (1 x outputSize) = (inputSize x outputSize)
        Matrix xt = lastInput.transpose();
        Matrix batchDW = xt.multiply(outputGradient);

        // dB = sum of dY across rows (1 x outputSize)
        Matrix batchDB = outputGradient.sumRows();

        // Accumulate gradients
        if (dW == null) {
            dW = Matrix.zeros(inputSize, outputSize);
        }
        if (dB == null) {
            dB = Matrix.zeros(1, outputSize);
        }
        dW.addInPlace(batchDW);
        dB.addInPlace(batchDB);
        gradientCount++;

        // dX = dY * W^T (1 x outputSize) * (outputSize x inputSize) = (1 x inputSize)
        Matrix wt = weights.transpose();
        return outputGradient.multiply(wt);
    }

    @Override
    public void updateWeights(double learningRate) {
        if (gradientCount == 0) return;

        // Average gradients if accumulated over multiple samples
        double scale = learningRate / gradientCount;

        // W = W - learningRate * dW
        Matrix scaledDW = dW.multiply(scale);
        this.weights = this.weights.subtract(scaledDW);

        // B = B - learningRate * dB
        Matrix scaledDB = dB.multiply(scale);
        this.bias = this.bias.subtract(scaledDB);

        // Reset accumulators
        this.dW.fill(0.0);
        this.dB.fill(0.0);
        this.gradientCount = 0;
    }

    @Override
    public boolean hasWeights() {
        return true;
    }

    public Matrix getWeights() {
        return weights;
    }

    public void setWeights(Matrix weights) {
        this.weights = weights;
    }

    public Matrix getBias() {
        return bias;
    }

    public void setBias(Matrix bias) {
        this.bias = bias;
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getOutputSize() {
        return outputSize;
    }
}