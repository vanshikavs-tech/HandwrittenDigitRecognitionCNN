package layers;

import math.Matrix;

public interface Layer {

    Matrix forward(Matrix input);

    Matrix backward(Matrix gradient);

}