package math;

import java.io.Serializable;
import java.util.Random;

/**
 * Tensor represents a 3-dimensional volume (channels x height x width).
 * Used for storing multi-channel images (e.g. 1x28x28 for MNIST grayscale),
 * convolutional feature maps, filter kernels, and pooling activations.
 */
public class Tensor implements Serializable {
    private static final long serialVersionUID = 1L;

    private final int channels;
    private final int height;
    private final int width;
    private final double[][][] data;

    private static final Random random = new Random();

    // Primary constructor
    public Tensor(int channels, int height, int width) {
        if (channels <= 0 || height <= 0 || width <= 0) {
            throw new IllegalArgumentException(
                "Tensor dimensions must be positive. Got: " + channels + "x" + height + "x" + width
            );
        }
        this.channels = channels;
        this.height = height;
        this.width = width;
        this.data = new double[channels][height][width];
    }

    // Constructor from 3D array (deep copy)
    public Tensor(double[][][] values) {
        if (values == null || values.length == 0 || values[0].length == 0 || values[0][0].length == 0) {
            throw new IllegalArgumentException("Input values must be a non-empty 3D array.");
        }
        this.channels = values.length;
        this.height = values[0].length;
        this.width = values[0][0].length;
        this.data = new double[channels][height][width];

        for (int c = 0; c < channels; c++) {
            if (values[c].length != height) {
                throw new IllegalArgumentException("Inconsistent height in 3D array at channel " + c);
            }
            for (int h = 0; h < height; h++) {
                if (values[c][h].length != width) {
                    throw new IllegalArgumentException("Inconsistent width in 3D array at channel " + c + ", row " + h);
                }
                System.arraycopy(values[c][h], 0, this.data[c][h], 0, width);
            }
        }
    }

    // Wrap a 2D Matrix as a 1-channel Tensor (1 x height x width)
    public Tensor(Matrix matrix) {
        this(1, matrix.getRows(), matrix.getCols());
        for (int r = 0; r < height; r++) {
            for (int col = 0; col < width; col++) {
                this.data[0][r][col] = matrix.getValue(r, col);
            }
        }
    }

    // Static factory for zeros
    public static Tensor zeros(int channels, int height, int width) {
        return new Tensor(channels, height, width);
    }

    // Static factory from 1D flat array
    public static Tensor fromFlatArray(double[] flat, int channels, int height, int width) {
        if (flat.length != channels * height * width) {
            throw new IllegalArgumentException(
                "Flat array length (" + flat.length + ") does not match volume (" +
                (channels * height * width) + ")"
            );
        }
        Tensor tensor = new Tensor(channels, height, width);
        int idx = 0;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    tensor.setValue(c, h, w, flat[idx++]);
                }
            }
        }
        return tensor;
    }

    // Static factory from Matrix
    public static Tensor fromMatrix(Matrix matrix) {
        return new Tensor(matrix);
    }

    // Get value at (channel, row, col)
    public double getValue(int channel, int row, int col) {
        checkBounds(channel, row, col);
        return data[channel][row][col];
    }

    // Set value at (channel, row, col)
    public void setValue(int channel, int row, int col, double value) {
        checkBounds(channel, row, col);
        data[channel][row][col] = value;
    }

    public int getChannels() {
        return channels;
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }

    public int size() {
        return channels * height * width;
    }

    // Extract a single 2D slice/channel as a Matrix (height x width)
    public Matrix getChannel(int channel) {
        if (channel < 0 || channel >= channels) {
            throw new IndexOutOfBoundsException("Channel " + channel + " out of bounds [0, " + channels + ")");
        }
        Matrix m = new Matrix(height, width);
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                m.setValue(h, w, data[channel][h][w]);
            }
        }
        return m;
    }

    // Replace a single channel with a Matrix
    public void setChannel(int channel, Matrix matrix) {
        if (channel < 0 || channel >= channels) {
            throw new IndexOutOfBoundsException("Channel " + channel + " out of bounds [0, " + channels + ")");
        }
        if (matrix.getRows() != height || matrix.getCols() != width) {
            throw new IllegalArgumentException(
                "Matrix size (" + matrix.getRows() + "x" + matrix.getCols() +
                ") does not match tensor channel size (" + height + "x" + width + ")"
            );
        }
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                data[channel][h][w] = matrix.getValue(h, w);
            }
        }
    }

    // Flatten to 1D double array
    public double[] toFlatArray() {
        double[] flat = new double[channels * height * width];
        int idx = 0;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    flat[idx++] = data[c][h][w];
                }
            }
        }
        return flat;
    }

    // Flatten into a 1 x (C*H*W) Matrix row vector
    public Matrix flattenToMatrix() {
        return Matrix.fromVector(toFlatArray());
    }

    // Convert to Matrix: if 1 channel, returns Matrix(H, W); otherwise flattens to 1 x (C*H*W)
    public Matrix toMatrix() {
        if (channels == 1) {
            return getChannel(0);
        }
        return flattenToMatrix();
    }

    // Deep copy of this Tensor
    public Tensor copy() {
        Tensor copy = new Tensor(channels, height, width);
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                System.arraycopy(this.data[c][h], 0, copy.data[c][h], 0, width);
            }
        }
        return copy;
    }

    // Fill all elements with a single value
    public void fill(double value) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    data[c][h][w] = value;
                }
            }
        }
    }

    // Uniform random initialization in [-0.5, 0.5]
    public void randomInitialize() {
        randomInitialize(-0.5, 0.5);
    }

    // Uniform random initialization in [min, max]
    public void randomInitialize(double min, double max) {
        double range = max - min;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    data[c][h][w] = min + random.nextDouble() * range;
                }
            }
        }
    }

    // He normal initialization for Conv kernels
    public void heInitialize(int fanIn) {
        double stdDev = Math.sqrt(2.0 / fanIn);
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    data[c][h][w] = random.nextGaussian() * stdDev;
                }
            }
        }
    }

    // Xavier uniform initialization
    public void xavierInitialize(int fanIn, int fanOut) {
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        randomInitialize(-limit, limit);
    }

    // Element-wise addition
    public Tensor add(Tensor other) {
        checkSameDimensions(other);
        Tensor result = new Tensor(channels, height, width);
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    result.data[c][h][w] = this.data[c][h][w] + other.data[c][h][w];
                }
            }
        }
        return result;
    }

    // In-place addition
    public void addInPlace(Tensor other) {
        checkSameDimensions(other);
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    this.data[c][h][w] += other.data[c][h][w];
                }
            }
        }
    }

    // Element-wise subtraction
    public Tensor subtract(Tensor other) {
        checkSameDimensions(other);
        Tensor result = new Tensor(channels, height, width);
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    result.data[c][h][w] = this.data[c][h][w] - other.data[c][h][w];
                }
            }
        }
        return result;
    }

    // Scalar multiplication
    public Tensor multiply(double scalar) {
        Tensor result = new Tensor(channels, height, width);
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    result.data[c][h][w] = this.data[c][h][w] * scalar;
                }
            }
        }
        return result;
    }

    // Hadamard (element-wise) multiplication
    public Tensor hadamard(Tensor other) {
        checkSameDimensions(other);
        Tensor result = new Tensor(channels, height, width);
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    result.data[c][h][w] = this.data[c][h][w] * other.data[c][h][w];
                }
            }
        }
        return result;
    }

    // Sum of all values
    public double sum() {
        double total = 0.0;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    total += data[c][h][w];
                }
            }
        }
        return total;
    }

    // Max value
    public double max() {
        double maxVal = -Double.MAX_VALUE;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    if (data[c][h][w] > maxVal) {
                        maxVal = data[c][h][w];
                    }
                }
            }
        }
        return maxVal;
    }

    // ArgMax index (flat order)
    public int argMax() {
        int bestIdx = 0;
        double maxVal = -Double.MAX_VALUE;
        int currentIdx = 0;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    if (data[c][h][w] > maxVal) {
                        maxVal = data[c][h][w];
                        bestIdx = currentIdx;
                    }
                    currentIdx++;
                }
            }
        }
        return bestIdx;
    }

    public void printDimensions() {
        System.out.println(channels + " x " + height + " x " + width);
    }

    public void printTensor() {
        for (int c = 0; c < channels; c++) {
            System.out.println("Channel " + c + " (" + height + "x" + width + "):");
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    System.out.printf("%8.4f ", data[c][h][w]);
                }
                System.out.println();
            }
            System.out.println();
        }
    }

    private void checkBounds(int channel, int row, int col) {
        if (channel < 0 || channel >= channels || row < 0 || row >= height || col < 0 || col >= width) {
            throw new IndexOutOfBoundsException(
                "Invalid index (" + channel + ", " + row + ", " + col + ") for Tensor of size " +
                channels + "x" + height + "x" + width
            );
        }
    }

    private void checkSameDimensions(Tensor other) {
        if (this.channels != other.channels || this.height != other.height || this.width != other.width) {
            throw new IllegalArgumentException(
                "Tensor dimensions must match: " +
                this.channels + "x" + this.height + "x" + this.width + " vs " +
                other.channels + "x" + other.height + "x" + other.width
            );
        }
    }
}