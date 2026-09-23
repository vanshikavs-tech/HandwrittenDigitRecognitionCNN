package math;

import java.io.Serializable;
import java.util.Random;

/**
 * Matrix represents a 2D mathematical matrix of double precision values.
 * Acts as the foundational linear algebra engine for dense layers,
 * weight matrices, bias vectors, and matrix transformations.
 */
public class Matrix implements Serializable {
    private static final long serialVersionUID = 1L;

    private final int rows;
    private final int cols;
    private final double[][] data;

    private static final Random random = new Random();

    // Primary constructor
    public Matrix(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("Matrix dimensions must be positive. Got: " + rows + "x" + cols);
        }
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    // Constructor from existing 2D array (deep copy)
    public Matrix(double[][] values) {
        if (values == null || values.length == 0 || values[0].length == 0) {
            throw new IllegalArgumentException("Input array must be non-empty.");
        }
        this.rows = values.length;
        this.cols = values[0].length;
        this.data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            if (values[i].length != cols) {
                throw new IllegalArgumentException("All rows must have the same number of columns.");
            }
            System.arraycopy(values[i], 0, this.data[i], 0, cols);
        }
    }

    // Static factory for zeros
    public static Matrix zeros(int rows, int cols) {
        return new Matrix(rows, cols);
    }

    // Static factory for a 1D vector as a 1 x N row matrix
    public static Matrix fromVector(double[] vector) {
        Matrix m = new Matrix(1, vector.length);
        for (int j = 0; j < vector.length; j++) {
            m.setValue(0, j, vector[j]);
        }
        return m;
    }

    // Get value at (row, col)
    public double getValue(int row, int col) {
        checkBounds(row, col);
        return data[row][col];
    }

    // Set value at (row, col)
    public void setValue(int row, int col, double value) {
        checkBounds(row, col);
        data[row][col] = value;
    }

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public double[][] toArray() {
        double[][] copy = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, copy[i], 0, cols);
        }
        return copy;
    }

    // Flatten to 1D array
    public double[] toFlatArray() {
        double[] flat = new double[rows * cols];
        int idx = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flat[idx++] = data[i][j];
            }
        }
        return flat;
    }

    // Deep copy of this matrix
    public Matrix copy() {
        Matrix copy = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            System.arraycopy(this.data[i], 0, copy.data[i], 0, cols);
        }
        return copy;
    }

    // Fill all elements with a single scalar value
    public void fill(double value) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = value;
            }
        }
    }

    // Standard uniform random initialization in [-0.5, 0.5]
    public void randomInitialize() {
        randomInitialize(-0.5, 0.5);
    }

    // Uniform random initialization in [min, max]
    public void randomInitialize(double min, double max) {
        double range = max - min;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = min + random.nextDouble() * range;
            }
        }
    }

    // Xavier/Glorot uniform initialization: suitable for Sigmoid, Softmax, Tanh
    public void xavierInitialize(int fanIn, int fanOut) {
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        randomInitialize(-limit, limit);
    }

    // He/Kaiming normal initialization: optimal for ReLU activations
    public void heInitialize(int fanIn) {
        double stdDev = Math.sqrt(2.0 / fanIn);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = random.nextGaussian() * stdDev;
            }
        }
    }

    // Matrix addition (A + B)
    public Matrix add(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException(
                "Matrix dimensions must match for addition: " +
                this.rows + "x" + this.cols + " vs " + other.rows + "x" + other.cols
            );
        }
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    // In-place addition (this += other)
    public void addInPlace(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix dimensions must match for in-place addition.");
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this.data[i][j] += other.data[i][j];
            }
        }
    }

    // Matrix subtraction (A - B)
    public Matrix subtract(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException(
                "Matrix dimensions must match for subtraction: " +
                this.rows + "x" + this.cols + " vs " + other.rows + "x" + other.cols
            );
        }
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    // Scalar multiplication (A * scalar)
    public Matrix multiply(double scalar) {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] * scalar;
            }
        }
        return result;
    }

    // Alias for scalar multiplication
    public Matrix scalarMultiply(double scalar) {
        return multiply(scalar);
    }

    // Matrix multiplication (this x other): (R1 x C1) * (C1 x C2) = (R1 x C2)
    public Matrix multiply(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException(
                "Matrix multiplication dimension mismatch: cannot multiply " +
                this.rows + "x" + this.cols + " with " + other.rows + "x" + other.cols
            );
        }
        Matrix result = new Matrix(this.rows, other.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int k = 0; k < this.cols; k++) {
                double a = this.data[i][k];
                if (a != 0.0) { // Optimization for sparse values
                    for (int j = 0; j < other.cols; j++) {
                        result.data[i][j] += a * other.data[k][j];
                    }
                }
            }
        }
        return result;
    }

    // Hadamard (element-wise) multiplication: (A .* B)
    public Matrix hadamard(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException(
                "Dimensions must match for Hadamard product: " +
                this.rows + "x" + this.cols + " vs " + other.rows + "x" + other.cols
            );
        }
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    // Transpose: A^T of size (cols x rows)
    public Matrix transpose() {
        Matrix result = new Matrix(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[j][i] = this.data[i][j];
            }
        }
        return result;
    }

    // Sum of all elements in the matrix
    public double sum() {
        double total = 0.0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                total += data[i][j];
            }
        }
        return total;
    }

    // Sum down the columns -> returns a 1 x cols row vector
    public Matrix sumRows() {
        Matrix result = new Matrix(1, cols);
        for (int j = 0; j < cols; j++) {
            double colSum = 0.0;
            for (int i = 0; i < rows; i++) {
                colSum += data[i][j];
            }
            result.data[0][j] = colSum;
        }
        return result;
    }

    // Maximum value across all elements
    public double max() {
        double maxVal = -Double.MAX_VALUE;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (data[i][j] > maxVal) {
                    maxVal = data[i][j];
                }
            }
        }
        return maxVal;
    }

    // Minimum value across all elements
    public double min() {
        double minVal = Double.MAX_VALUE;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (data[i][j] < minVal) {
                    minVal = data[i][j];
                }
            }
        }
        return minVal;
    }

    // ArgMax: returns the index of the highest element (for 1xN vectors, returns the column index)
    public int argMax() {
        int bestIdx = 0;
        double maxVal = -Double.MAX_VALUE;
        int currentIdx = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (data[i][j] > maxVal) {
                    maxVal = data[i][j];
                    bestIdx = currentIdx;
                }
                currentIdx++;
            }
        }
        return bestIdx;
    }

    // Convert 2D Matrix to 3D Tensor of shape (channels, height, width)
    public Tensor toTensor(int channels, int height, int width) {
        if (channels * height * width != this.rows * this.cols) {
            throw new IllegalArgumentException(
                "Total elements (" + (rows * cols) + ") must match tensor volume (" +
                (channels * height * width) + ")"
            );
        }
        Tensor tensor = new Tensor(channels, height, width);
        int idx = 0;
        double[] flat = toFlatArray();
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    tensor.setValue(c, h, w, flat[idx++]);
                }
            }
        }
        return tensor;
    }

    public void printDimensions() {
        System.out.println(rows + " x " + cols);
    }

    public void printMatrix() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.printf("%8.4f ", data[i][j]);
            }
            System.out.println();
        }
    }

    private void checkBounds(int row, int col) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw new IndexOutOfBoundsException(
                "Invalid index (" + row + ", " + col + ") for Matrix of size " + rows + "x" + cols
            );
        }
    }
}