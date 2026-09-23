package layers;

import math.Tensor;

/**
 * MaxPoolingLayer performs 2D spatial downsampling across each channel independently.
 * Reduces dimensionality, retains dominant features, and provides translation invariance.
 *
 * Forward pass: takes maximum value within each poolSize x poolSize window.
 * Backward pass: routes the gradient exclusively to the location of the maximum element (argmax).
 */
public class MaxPoolingLayer implements Layer {
    private static final long serialVersionUID = 1L;

    private final int poolSize;
    private final int stride;

    // Cache dimensions and argmax locations for backpropagation
    private transient int cachedInChannels;
    private transient int cachedInHeight;
    private transient int cachedInWidth;
    private transient int[][][] argmaxH;
    private transient int[][][] argmaxW;

    public MaxPoolingLayer() {
        this(2, 2);
    }

    public MaxPoolingLayer(int poolSize, int stride) {
        if (poolSize <= 0 || stride <= 0) {
            throw new IllegalArgumentException("poolSize and stride must be positive.");
        }
        this.poolSize = poolSize;
        this.stride = stride;
    }

    @Override
    public Tensor forward(Tensor input) {
        this.cachedInChannels = input.getChannels();
        this.cachedInHeight = input.getHeight();
        this.cachedInWidth = input.getWidth();

        int outHeight = (cachedInHeight - poolSize) / stride + 1;
        int outWidth = (cachedInWidth - poolSize) / stride + 1;

        if (outHeight <= 0 || outWidth <= 0) {
            throw new IllegalArgumentException(
                "Input dimension (" + cachedInHeight + "x" + cachedInWidth +
                ") too small for poolSize " + poolSize + " with stride " + stride
            );
        }

        Tensor output = new Tensor(cachedInChannels, outHeight, outWidth);
        this.argmaxH = new int[cachedInChannels][outHeight][outWidth];
        this.argmaxW = new int[cachedInChannels][outHeight][outWidth];

        for (int c = 0; c < cachedInChannels; c++) {
            for (int oh = 0; oh < outHeight; oh++) {
                int startH = oh * stride;
                for (int ow = 0; ow < outWidth; ow++) {
                    int startW = ow * stride;

                    double maxVal = -Double.MAX_VALUE;
                    int bestH = startH;
                    int bestW = startW;

                    for (int ph = 0; ph < poolSize; ph++) {
                        int ih = startH + ph;
                        for (int pw = 0; pw < poolSize; pw++) {
                            int iw = startW + pw;
                            double val = input.getValue(c, ih, iw);
                            if (val > maxVal) {
                                maxVal = val;
                                bestH = ih;
                                bestW = iw;
                            }
                        }
                    }

                    output.setValue(c, oh, ow, maxVal);
                    argmaxH[c][oh][ow] = bestH;
                    argmaxW[c][oh][ow] = bestW;
                }
            }
        }

        return output;
    }

    @Override
    public Tensor backward(Tensor outputGradient) {
        if (argmaxH == null || argmaxW == null) {
            throw new IllegalStateException("Cannot run backward before forward pass.");
        }

        Tensor inputGradient = Tensor.zeros(cachedInChannels, cachedInHeight, cachedInWidth);
        int outHeight = outputGradient.getHeight();
        int outWidth = outputGradient.getWidth();

        for (int c = 0; c < cachedInChannels; c++) {
            for (int oh = 0; oh < outHeight; oh++) {
                for (int ow = 0; ow < outWidth; ow++) {
                    double grad = outputGradient.getValue(c, oh, ow);
                    int bh = argmaxH[c][oh][ow];
                    int bw = argmaxW[c][oh][ow];

                    // Route gradient to argmax position
                    double current = inputGradient.getValue(c, bh, bw);
                    inputGradient.setValue(c, bh, bw, current + grad);
                }
            }
        }

        return inputGradient;
    }

    public int getPoolSize() {
        return poolSize;
    }

    public int getStride() {
        return stride;
    }
}
