package loss;

import math.Matrix;
import math.Tensor;

/**
 * Categorical Cross-Entropy Loss:
 *   L = -sum_i ( y_i * log(p_i + eps) )
 *
 * For Softmax output, the combined gradient w.r.t. pre-softmax logits Z is simply:
 *   dL/dZ = (P - Y)
 * which is elegant, simple, and numerically robust.
 */
public class CrossEntropyLoss implements LossFunction {
    private static final long serialVersionUID = 1L;
    private static final double EPSILON = 1e-15;

    @Override
    public double calculateLoss(Tensor prediction, Tensor target) {
        return calculateLoss(prediction.toMatrix(), target.toMatrix());
    }

    @Override
    public Tensor calculateGradient(Tensor prediction, Tensor target) {
        Matrix grad = calculateGradient(prediction.toMatrix(), target.toMatrix());
        return Tensor.fromMatrix(grad);
    }

    @Override
    public double calculateLoss(Matrix prediction, Matrix target) {
        if (prediction.getCols() != target.getCols()) {
            throw new IllegalArgumentException(
                "Prediction and target dimension mismatch: " +
                prediction.getCols() + " vs " + target.getCols()
            );
        }

        double loss = 0.0;
        int cols = prediction.getCols();
        for (int j = 0; j < cols; j++) {
            double p = prediction.getValue(0, j);
            double y = target.getValue(0, j);
            // Clip p to prevent log(0)
            double clippedP = Math.max(EPSILON, Math.min(1.0 - EPSILON, p));
            loss -= y * Math.log(clippedP);
        }
        return loss;
    }

    /**
     * Computes the combined Softmax + Cross-Entropy gradient: (P - Y)
     */
    @Override
    public Matrix calculateGradient(Matrix prediction, Matrix target) {
        if (prediction.getCols() != target.getCols()) {
            throw new IllegalArgumentException(
                "Prediction and target dimension mismatch: " +
                prediction.getCols() + " vs " + target.getCols()
            );
        }

        int cols = prediction.getCols();
        Matrix gradient = new Matrix(1, cols);
        for (int j = 0; j < cols; j++) {
            double p = prediction.getValue(0, j);
            double y = target.getValue(0, j);
            gradient.setValue(0, j, p - y);
        }
        return gradient;
    }
}
