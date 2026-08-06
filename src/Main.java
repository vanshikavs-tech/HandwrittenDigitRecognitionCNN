import layers.FullyConnectedLayer;
import layers.ReLULayer;
import layers.SoftmaxLayer;
import math.Matrix;
import network.NeuralNetwork;

public class Main {

    public static void main(String[] args) {

        Matrix input = new Matrix(1,3);

        input.setValue(0,0,2);
        input.setValue(0,1,4);
        input.setValue(0,2,6);

        NeuralNetwork network = new NeuralNetwork();

        network.addLayer(new FullyConnectedLayer(3,4));

        network.addLayer(new ReLULayer());

        network.addLayer(new FullyConnectedLayer(4,3));

        network.addLayer(new SoftmaxLayer());

        Matrix prediction = network.predict(input);

        System.out.println("Prediction");

        prediction.printMatrix();

    }

}