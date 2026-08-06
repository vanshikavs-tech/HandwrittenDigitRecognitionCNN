package layers;

import math.Matrix;

public class ReLULayer implements Layer {

    @Override
    public Matrix forward(Matrix input) {

        Matrix output = new Matrix(input.getRows(), input.getCols());

        for (int i = 0; i < input.getRows(); i++) {

            for (int j = 0; j < input.getCols(); j++) {

                double value = input.getValue(i, j);

                if (value > 0)
                    output.setValue(i, j, value);
                else
                    output.setValue(i, j, 0);

            }

        }

        return output;

    }

    @Override
    public Matrix backward(Matrix gradient) {

        return gradient;

    }

}