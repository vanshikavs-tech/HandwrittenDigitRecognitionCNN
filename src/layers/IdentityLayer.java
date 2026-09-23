package layers;

import math.Matrix;
import math.Tensor;

/**
 * Identity Layer passes the input unchanged during the forward pass
 * and passes the gradient unchanged during the backward pass.
 * Useful for pipeline testing and verification.
 */
public class IdentityLayer implements Layer {
    private static final long serialVersionUID = 1L;

    @Override
    public Tensor forward(Tensor input) {
        return input;
    }

    @Override
    public Tensor backward(Tensor outputGradient) {
        return outputGradient;
    }

    public Matrix forward(Matrix input) {
        return input;
    }

    public Matrix backward(Matrix gradient) {
        return gradient;
    }
}