import layers.FullyConnectedLayer;
import layers.ReLULayer;
import layers.SoftmaxLayer;
import math.Matrix;
import network.NeuralNetwork;
import trainer.Trainer;

public class Main {

    public static void main(String[] args) {

        Matrix input = new Matrix(1,3);

        input.setValue(0,0,2);
        input.setValue(0,1,4);
        input.setValue(0,2,6);

        Matrix target = new Matrix(1,3);

        target.setValue(0,0,0);
        target.setValue(0,1,1);
        target.setValue(0,2,0);

        NeuralNetwork network = new NeuralNetwork();

        network.addLayer(new FullyConnectedLayer(3,4));
        network.addLayer(new ReLULayer());
        network.addLayer(new FullyConnectedLayer(4,3));
        network.addLayer(new SoftmaxLayer());

        Trainer trainer = new Trainer(network);

        trainer.train(input, target);

    }
}