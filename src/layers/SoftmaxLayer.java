package layers;

import math.Matrix;
import math.Tensor;

/**
 * Softmax Activation Layer.
 * Converts unnormalized logits into normalized probability distribution (sum = 1.0).
 * Implemented with numerical stability trick (subtracting max logit).
 * Backpropagation computes exact Jacobian vector product: dZ_i = P_i * (dY_i - sum(dY * P)).
 */
public class SoftmaxLayer implements Layer {
    private static final long serialVersionUID = 1L;

    private transient Tensor lastOutputTensor;
    private transient Matrix lastOutputMatrix;

    @Override
    public Tensor forward(Tensor input) {
        Matrix inMatrix = (input.getChannels() == 1 && input.getHeight() == 1)
                ? input.toMatrix()
                : input.flattenToMatrix();

        Matrix outMatrix = forward(inMatrix);
        this.lastOutputTensor = Tensor.fromMatrix(outMatrix);
        return this.lastOutputTensor;
    }

    public Matrix forward(Matrix input) {
        int cols = input.getCols();
        Matrix output = new Matrix(1, cols);

        // Find max for numerical stability (prevents overflow in Math.exp)
        double maxLogit = -Double.MAX_VALUE;
        for (int j = 0; j < cols; j++) {
            double val = input.getValue(0, j);
            if (val > maxLogit) {
                maxLogit = val;
            }
        }

        // Compute exponents and sum
        double sumExp = 0.0;
        double[] expVals = new double[cols];
        for (int j = 0; j < cols; j++) {
            expVals[j] = Math.exp(input.getValue(0, j) - maxLogit);
            sumExp += expVals[j];
        }

        // Normalize to probabilities
        for (int j = 0; j < cols; j++) {
            output.setValue(0, j, expVals[j] / sumExp);
        }

        this.lastOutputMatrix = output.copy();
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
     * Backward pass computing the Jacobian product:
     * dX_i = P_i * (dY_i - sum_k(dY_k * P_k))
     */
    public Matrix backward(Matrix outputGradient) {
        if (lastOutputMatrix == null) {
            throw new IllegalStateException("Cannot run backward before forward pass.");
        }

        int cols = lastOutputMatrix.getCols();
        Matrix inputGradient = new Matrix(1, cols);

        // Compute dot product of outputGradient and softmax probabilities: sum_k (dY_k * P_k)
        double dot = 0.0;
        for (int j = 0; j < cols; j++) {
            dot += outputGradient.getValue(0, j) * lastOutputMatrix.getValue(0, j);
        }

        // dX_i = P_i * (dY_i - dot)
        for (int j = 0; j < cols; j++) {
            double p = lastOutputMatrix.getValue(0, j);
            double dy = outputGradient.getValue(0, j);
            inputGradient.setValue(0, j, p * (dy - dot));
        }

        return inputGradient;
    }

    public Matrix getLastProbabilities() {
        return lastOutputMatrix;
    }
}