package layers;

import math.Matrix;
import math.Tensor;

/**
 * FlattenLayer transforms a 3D Tensor (channels x height x width)
 * into a flattened 1D vector (1 x 1 x C*H*W) or a 2D Matrix (1 x C*H*W)
 * to bridge the convolutional/pooling stages to the fully connected dense layers.
 * The backward pass reshapes the flat gradient vector back to the original 3D volume.
 */
public class FlattenLayer implements Layer {
    private static final long serialVersionUID = 1L;

    private transient int cachedChannels;
    private transient int cachedHeight;
    private transient int cachedWidth;

    @Override
    public Tensor forward(Tensor input) {
        this.cachedChannels = input.getChannels();
        this.cachedHeight = input.getHeight();
        this.cachedWidth = input.getWidth();

        double[] flat = input.toFlatArray();
        return Tensor.fromFlatArray(flat, 1, 1, flat.length);
    }

    public Matrix forwardToMatrix(Tensor input) {
        this.cachedChannels = input.getChannels();
        this.cachedHeight = input.getHeight();
        this.cachedWidth = input.getWidth();

        return input.flattenToMatrix();
    }

    @Override
    public Tensor backward(Tensor outputGradient) {
        if (cachedChannels == 0 || cachedHeight == 0 || cachedWidth == 0) {
            throw new IllegalStateException("Cannot run backward before forward pass.");
        }

        double[] flatGrad = outputGradient.toFlatArray();
        return Tensor.fromFlatArray(flatGrad, cachedChannels, cachedHeight, cachedWidth);
    }

    public Tensor backwardFromMatrix(Matrix outputGradient) {
        if (cachedChannels == 0 || cachedHeight == 0 || cachedWidth == 0) {
            throw new IllegalStateException("Cannot run backward before forward pass.");
        }

        double[] flatGrad = outputGradient.toFlatArray();
        return Tensor.fromFlatArray(flatGrad, cachedChannels, cachedHeight, cachedWidth);
    }

    public int getFlattenedSize() {
        return cachedChannels * cachedHeight * cachedWidth;
    }
}
