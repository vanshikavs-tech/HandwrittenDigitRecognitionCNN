package layers;

import java.util.Random;
import math.Tensor;

/**
 * ConvolutionLayer applies learnable 2D convolution filters across input channels.
 * Detects local spatial features (edges, corners, lines, loops).
 *
 * Implements forward convolution with padding and stride, and exact analytical backpropagation
 * for filter gradients (dFilters), bias gradients (dBiases), and input gradients (dX).
 */
public class ConvolutionLayer implements Layer {
    private static final long serialVersionUID = 1L;

    private final int inChannels;
    private final int outChannels;
    private final int kernelSize;
    private final int stride;
    private final int padding;

    // Parameters: filters[filterIndex][channelIndex][kernelRow][kernelCol]
    private double[][][][] filters;
    private double[] biases;

    // Gradient accumulators
    private double[][][][] dFilters;
    private double[] dBiases;
    private int gradientCount;

    // Cache from forward pass
    private transient Tensor lastInput;

    public ConvolutionLayer(int inChannels, int outChannels, int kernelSize) {
        this(inChannels, outChannels, kernelSize, 1, 0);
    }

    public ConvolutionLayer(int inChannels, int outChannels, int kernelSize, int stride, int padding) {
        if (inChannels <= 0 || outChannels <= 0 || kernelSize <= 0 || stride <= 0 || padding < 0) {
            throw new IllegalArgumentException("Invalid ConvolutionLayer parameters.");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;

        this.filters = new double[outChannels][inChannels][kernelSize][kernelSize];
        this.biases = new double[outChannels];
        this.dFilters = new double[outChannels][inChannels][kernelSize][kernelSize];
        this.dBiases = new double[outChannels];
        this.gradientCount = 0;

        initializeWeights();
    }

    private void initializeWeights() {
        // He (Kaiming) normal initialization
        int fanIn = inChannels * kernelSize * kernelSize;
        double stdDev = Math.sqrt(2.0 / fanIn);
        Random rand = new Random();

        for (int f = 0; f < outChannels; f++) {
            biases[f] = 0.0;
            for (int c = 0; c < inChannels; c++) {
                for (int kh = 0; kh < kernelSize; kh++) {
                    for (int kw = 0; kw < kernelSize; kw++) {
                        filters[f][c][kh][kw] = rand.nextGaussian() * stdDev;
                    }
                }
            }
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        if (input.getChannels() != inChannels) {
            throw new IllegalArgumentException(
                "Input channel mismatch: expected " + inChannels + ", got " + input.getChannels()
            );
        }
        this.lastInput = input.copy();

        int inHeight = input.getHeight();
        int inWidth = input.getWidth();

        int outHeight = (inHeight - kernelSize + 2 * padding) / stride + 1;
        int outWidth = (inWidth - kernelSize + 2 * padding) / stride + 1;

        if (outHeight <= 0 || outWidth <= 0) {
            throw new IllegalArgumentException(
                "Input dimensions (" + inHeight + "x" + inWidth + ") too small for kernel size " + kernelSize
            );
        }

        Tensor output = new Tensor(outChannels, outHeight, outWidth);

        for (int f = 0; f < outChannels; f++) {
            double biasVal = biases[f];
            for (int oh = 0; oh < outHeight; oh++) {
                int baseH = oh * stride - padding;
                for (int ow = 0; ow < outWidth; ow++) {
                    int baseW = ow * stride - padding;

                    double sum = biasVal;
                    for (int c = 0; c < inChannels; c++) {
                        for (int kh = 0; kh < kernelSize; kh++) {
                            int ih = baseH + kh;
                            if (ih < 0 || ih >= inHeight) continue;

                            for (int kw = 0; kw < kernelSize; kw++) {
                                int iw = baseW + kw;
                                if (iw < 0 || iw >= inWidth) continue;

                                sum += input.getValue(c, ih, iw) * filters[f][c][kh][kw];
                            }
                        }
                    }
                    output.setValue(f, oh, ow, sum);
                }
            }
        }

        return output;
    }

    @Override
    public Tensor backward(Tensor outputGradient) {
        if (lastInput == null) {
            throw new IllegalStateException("Cannot run backward before forward pass.");
        }

        int inHeight = lastInput.getHeight();
        int inWidth = lastInput.getWidth();
        int outHeight = outputGradient.getHeight();
        int outWidth = outputGradient.getWidth();

        Tensor inputGradient = Tensor.zeros(inChannels, inHeight, inWidth);

        // Ensure gradient accumulators exist
        if (dFilters == null) {
            dFilters = new double[outChannels][inChannels][kernelSize][kernelSize];
            dBiases = new double[outChannels];
        }

        for (int f = 0; f < outChannels; f++) {
            for (int oh = 0; oh < outHeight; oh++) {
                int baseH = oh * stride - padding;
                for (int ow = 0; ow < outWidth; ow++) {
                    int baseW = ow * stride - padding;
                    double grad = outputGradient.getValue(f, oh, ow);

                    // Bias gradient: dL/dB_f = sum(dL/dY_f)
                    dBiases[f] += grad;

                    for (int c = 0; c < inChannels; c++) {
                        for (int kh = 0; kh < kernelSize; kh++) {
                            int ih = baseH + kh;
                            if (ih < 0 || ih >= inHeight) continue;

                            for (int kw = 0; kw < kernelSize; kw++) {
                                int iw = baseW + kw;
                                if (iw < 0 || iw >= inWidth) continue;

                                // Filter gradient: dL/dK = grad * input
                                dFilters[f][c][kh][kw] += grad * lastInput.getValue(c, ih, iw);

                                // Input gradient: dL/dX = grad * filter
                                double currentGrad = inputGradient.getValue(c, ih, iw);
                                inputGradient.setValue(c, ih, iw, currentGrad + grad * filters[f][c][kh][kw]);
                            }
                        }
                    }
                }
            }
        }

        gradientCount++;
        return inputGradient;
    }

    @Override
    public void updateWeights(double learningRate) {
        if (gradientCount == 0) return;

        double scale = learningRate / gradientCount;

        for (int f = 0; f < outChannels; f++) {
            biases[f] -= scale * dBiases[f];
            dBiases[f] = 0.0;

            for (int c = 0; c < inChannels; c++) {
                for (int kh = 0; kh < kernelSize; kh++) {
                    for (int kw = 0; kw < kernelSize; kw++) {
                        filters[f][c][kh][kw] -= scale * dFilters[f][c][kh][kw];
                        dFilters[f][c][kh][kw] = 0.0;
                    }
                }
            }
        }

        gradientCount = 0;
    }

    @Override
    public boolean hasWeights() {
        return true;
    }

    public int getInChannels() {
        return inChannels;
    }

    public int getOutChannels() {
        return outChannels;
    }

    public int getKernelSize() {
        return kernelSize;
    }

    public double[][][][] getFilters() {
        return filters;
    }

    public double[] getBiases() {
        return biases;
    }
}
