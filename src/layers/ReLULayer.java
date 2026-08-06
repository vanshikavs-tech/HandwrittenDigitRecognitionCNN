package layers;

import math.Matrix;

public class ReLULayer implements Layer {

    private Matrix input;

    @Override
    public Matrix forward(Matrix input) {

        this.input = input;

        Matrix output = new Matrix(input.getRows(), input.getCols());

        for (int i = 0; i < input.getRows(); i++) {

            for (int j = 0; j < input.getCols(); j++) {

                double value = input.getValue(i, j);

                output.setValue(i, j, Math.max(0, value));
            }
        }

        return output;
    }

    @Override
    public Matrix backward(Matrix gradient) {

        Matrix outputGradient =
                new Matrix(input.getRows(), input.getCols());

        for (int i = 0; i < input.getRows(); i++) {

            for (int j = 0; j < input.getCols(); j++) {

                if (input.getValue(i, j) > 0)
                    outputGradient.setValue(i, j,
                            gradient.getValue(i, j));
                else
                    outputGradient.setValue(i, j, 0);

            }

        }

        return outputGradient;
    }

    @Override
    public void updateWeights(double learningRate) {
        // ReLU has no weights
    }
}