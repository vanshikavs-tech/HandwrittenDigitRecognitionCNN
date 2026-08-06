package math;

public class Matrix {

    private int rows;
    private int cols;
    private double[][] data;

    // Constructor
    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        data = new double[rows][cols];
    }

    // Set value at a specific position
    public void setValue(int row, int col, double value) {
        data[row][col] = value;
    }

    // Get value from a specific position
    public double getValue(int row, int col) {
        return data[row][col];
    }

    // Print the matrix
    public void printMatrix() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.print(data[i][j] + " ");
            }
            System.out.println();
        }
    }

    // Add two matrices
    public Matrix add(Matrix other) {

        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix dimensions must match.");
        }

        Matrix result = new Matrix(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.setValue(i, j, this.getValue(i, j) + other.getValue(i, j));
            }
        }

        return result;
    }

    // Subtract two matrices
public Matrix subtract(Matrix other) {

    if (this.rows != other.rows || this.cols != other.cols) {
        throw new IllegalArgumentException("Matrix dimensions must match.");
    }

    Matrix result = new Matrix(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.setValue(i, j, this.getValue(i, j) - other.getValue(i, j));
        }
    }

    return result;
}
   
// Multiply matrix by a scalar
public Matrix multiply(double scalar) {

    Matrix result = new Matrix(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.setValue(i, j, this.getValue(i, j) * scalar);
        }
    }

    return result;
}

// Matrix multiplication
public Matrix multiply(Matrix other) {

    if (this.cols != other.rows) {
        throw new IllegalArgumentException("Matrix multiplication not possible.");
    }

    Matrix result = new Matrix(this.rows, other.cols);

    for (int i = 0; i < this.rows; i++) {

        for (int j = 0; j < other.cols; j++) {

            double sum = 0;

            for (int k = 0; k < this.cols; k++) {

                sum += this.getValue(i, k) * other.getValue(k, j);

            }

            result.setValue(i, j, sum);

        }

    }

    return result;
}

public void printDimensions() {
    System.out.println(rows + " x " + cols);
}

public void randomInitialize() {

    for (int i = 0; i < rows; i++) {

        for (int j = 0; j < cols; j++) {

            data[i][j] = Math.random() - 0.5;

        }

    }

}

public int getRows() {
    return rows;
}

public int getCols() {
    return cols;
}

public void printDimensions() {
    System.out.println(rows + " x " + cols);
}

}