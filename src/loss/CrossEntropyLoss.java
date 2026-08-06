package loss;

import math.Matrix;

public class CrossEntropyLoss {

    public double calculateLoss(Matrix prediction, Matrix target) {

        double loss = 0.0;

        for (int j = 0; j < prediction.getCols(); j++) {

            double p = prediction.getValue(0, j);
            double t = target.getValue(0, j);

            if (p > 0) {
                loss -= t * Math.log(p);
            }
        }

        return loss;
    }
}