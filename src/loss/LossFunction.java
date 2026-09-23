package loss;

import java.io.Serializable;
import math.Matrix;
import math.Tensor;

/**
 * Interface for neural network loss functions.
 * Computes scalar loss (measuring prediction error)
 * and gradient with respect to network predictions (for backpropagation).
 */
public interface LossFunction extends Serializable {

    /**
     * Computes scalar loss value.
     *
     * @param prediction network output probabilities/predictions
     * @param target ground truth one-hot vector
     * @return scalar loss
     */
    double calculateLoss(Tensor prediction, Tensor target);

    /**
     * Computes gradient of loss with respect to predictions dL/dP.
     *
     * @param prediction network output probabilities/predictions
     * @param target ground truth one-hot vector
     * @return gradient tensor
     */
    Tensor calculateGradient(Tensor prediction, Tensor target);

    double calculateLoss(Matrix prediction, Matrix target);

    Matrix calculateGradient(Matrix prediction, Matrix target);
}
