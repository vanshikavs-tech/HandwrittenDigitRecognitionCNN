package layers;

import math.Matrix;

public class SoftmaxLayer implements Layer {

    @Override
    public Matrix forward(Matrix input) {

        Matrix output =
                new Matrix(input.getRows(), input.getCols());

        double sum = 0;

        for (int j = 0; j < input.getCols(); j++) {
            sum += Math.exp(input.getValue(0, j));
        }

        for (int j = 0; j < input.getCols(); j++) {

            double probability =
                    Math.exp(input.getValue(0, j)) / sum;

            output.setValue(0, j, probability);

        }

        return output;
    }

    @Override
    public Matrix backward(Matrix gradient) {

        return gradient;

    }

    @Override
    public void updateWeights(double learningRate) {

    }
}