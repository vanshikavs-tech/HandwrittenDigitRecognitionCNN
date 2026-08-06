package trainer;

import math.Matrix;
import network.NeuralNetwork;
import loss.CrossEntropyLoss;

public class Trainer {

    private NeuralNetwork network;
    private CrossEntropyLoss lossFunction;

    public Trainer(NeuralNetwork network) {
        this.network = network;
        this.lossFunction = new CrossEntropyLoss();
    }

    public void train(Matrix input, Matrix target) {

        Matrix prediction = network.predict(input);

        double loss = lossFunction.calculateLoss(prediction, target);

        System.out.println("Prediction:");
        prediction.printMatrix();

        System.out.println();

        System.out.println("Loss = " + loss);

        // Backpropagation will be added later
    }
}