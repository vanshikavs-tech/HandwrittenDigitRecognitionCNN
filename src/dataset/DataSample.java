package dataset;

import java.io.Serializable;
import math.Matrix;
import math.Tensor;

/**
 * DataSample encapsulates a single labeled training or test observation.
 * Pairs an input image Tensor with its integer class label and one-hot target Tensor.
 */
public class DataSample implements Serializable {
    private static final long serialVersionUID = 1L;

    private final Tensor image;
    private final int label;
    private final Tensor target;

    public DataSample(Tensor image, int label) {
        this(image, label, 10);
    }

    public DataSample(Tensor image, int label, int numClasses) {
        this.image = image;
        this.label = label;

        // Generate one-hot encoded target: [0, 0, ..., 1, ..., 0]
        Matrix targetMatrix = Matrix.zeros(1, numClasses);
        if (label >= 0 && label < numClasses) {
            targetMatrix.setValue(0, label, 1.0);
        }
        this.target = Tensor.fromMatrix(targetMatrix);
    }

    public Tensor getImage() {
        return image;
    }

    public int getLabel() {
        return label;
    }

    public Tensor getTarget() {
        return target;
    }

    public Matrix getTargetMatrix() {
        return target.toMatrix();
    }
}
