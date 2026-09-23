package layers;

import math.Matrix;
import math.Tensor;

/**
 * Rectified Linear Unit (ReLU) Activation Layer.
 * Forward: Y = max(0, X)
 * Backward: dX = dY * (X > 0 ? 1 : 0)
 * Introduces non-linearity, enabling the network to learn complex non-linear patterns.
 */
public class ReLULayer implements Layer {
    private static final long serialVersionUID = 1L;

    private transient Tensor lastInputTensor;
    private transient Matrix lastInputMatrix;

    @Override
    public Tensor forward(Tensor input) {
        this.lastInputTensor = input.copy();
        Tensor output = new Tensor(input.getChannels(), input.getHeight(), input.getWidth());

        for (int c = 0; c < input.getChannels(); c++) {
            for (int h = 0; h < input.getHeight(); h++) {
                for (int w = 0; w < input.getWidth(); w++) {
                    double val = input.getValue(c, h, w);
                    output.setValue(c, h, w, val > 0 ? val : 0.0);
                }
            }
        }
        return output;
    }

    public Matrix forward(Matrix input) {
        this.lastInputMatrix = input.copy();
        Matrix output = new Matrix(input.getRows(), input.getCols());

        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getCols(); j++) {
                double val = input.getValue(i, j);
                output.setValue(i, j, val > 0 ? val : 0.0);
            }
        }
        return output;
    }

    @Override
    public Tensor backward(Tensor outputGradient) {
        if (lastInputTensor == null) {
            // Fallback if matrix forward was invoked
            if (lastInputMatrix != null) {
                Matrix inGrad = backward(outputGradient.toMatrix());
                return Tensor.fromMatrix(inGrad);
            }
            throw new IllegalStateException("Cannot run backward before forward pass.");
        }

        Tensor inputGradient = new Tensor(
            outputGradient.getChannels(),
            outputGradient.getHeight(),
            outputGradient.getWidth()
        );

        for (int c = 0; c < outputGradient.getChannels(); c++) {
            for (int h = 0; h < outputGradient.getHeight(); h++) {
                for (int w = 0; w < outputGradient.getWidth(); w++) {
                    double inVal = lastInputTensor.getValue(c, h, w);
                    double outGrad = outputGradient.getValue(c, h, w);
                    inputGradient.setValue(c, h, w, inVal > 0 ? outGrad : 0.0);
                }
            }
        }
        return inputGradient;
    }

    public Matrix backward(Matrix outputGradient) {
        if (lastInputMatrix == null) {
            throw new IllegalStateException("Cannot run backward before forward pass.");
        }

        Matrix inputGradient = new Matrix(outputGradient.getRows(), outputGradient.getCols());
        for (int i = 0; i < outputGradient.getRows(); i++) {
            for (int j = 0; j < outputGradient.getCols(); j++) {
                double inVal = lastInputMatrix.getValue(i, j);
                double outGrad = outputGradient.getValue(i, j);
                inputGradient.setValue(i, j, inVal > 0 ? outGrad : 0.0);
            }
        }
        return inputGradient;
    }
}