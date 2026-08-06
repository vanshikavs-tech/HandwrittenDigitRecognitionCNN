package math;

public class Tensor {

    private int channels;
    private int height;
    private int width;
    private double[][][] data;

    // Constructor
    public Tensor(int channels, int height, int width) {
        this.channels = channels;
        this.height = height;
        this.width = width;

        data = new double[channels][height][width];
    }

    // Set value
    public void setValue(int channel, int row, int col, double value) {
        data[channel][row][col] = value;
    }

    // Get value
    public double getValue(int channel, int row, int col) {
        return data[channel][row][col];
    }

    // Get number of channels
    public int getChannels() {
        return channels;
    }

    // Get height
    public int getHeight() {
        return height;
    }

    // Get width
    public int getWidth() {
        return width;
    }

    // Fill entire tensor with one value
    public void fill(double value) {
        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    data[c][i][j] = value;
                }
            }
        }
    }

    // Random initialization
    public void randomInitialize() {
        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    data[c][i][j] = Math.random() - 0.5;
                }
            }
        }
    }

    // Print tensor
    public void printTensor() {
        for (int c = 0; c < channels; c++) {
            System.out.println("Channel " + c);

            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    System.out.print(data[c][i][j] + " ");
                }
                System.out.println();
            }

            System.out.println();
        }
    }
}