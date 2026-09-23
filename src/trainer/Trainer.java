package trainer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import dataset.DataSample;
import loss.CrossEntropyLoss;
import loss.LossFunction;
import math.Tensor;
import network.NeuralNetwork;

/**
 * Trainer engine coordinates the training loop, forward pass, loss calculation,
 * backpropagation, mini-batch gradient descent, and validation evaluation.
 */
public class Trainer {

    private final NeuralNetwork network;
    private final LossFunction lossFunction;

    public Trainer(NeuralNetwork network) {
        this(network, new CrossEntropyLoss());
    }

    public Trainer(NeuralNetwork network, LossFunction lossFunction) {
        this.network = network;
        this.lossFunction = lossFunction;
    }

    /**
     * Trains the network on a list of labeled DataSamples using mini-batch gradient descent.
     *
     * @param trainData    training samples
     * @param epochs       number of complete passes over the dataset
     * @param learningRate gradient descent learning rate (e.g. 0.01)
     * @param batchSize    mini-batch size (e.g. 32 or 1 for pure SGD)
     * @param callback     optional listener for progress updates
     */
    public void train(
        List<DataSample> trainData,
        int epochs,
        double learningRate,
        int batchSize,
        TrainingCallback callback
    ) {
        if (trainData == null || trainData.isEmpty()) {
            throw new IllegalArgumentException("Training dataset cannot be empty.");
        }

        List<DataSample> dataset = new ArrayList<>(trainData);
        int totalSamples = dataset.size();
        int totalBatches = (int) Math.ceil((double) totalSamples / batchSize);

        System.out.printf(
            "%n[Training Started] Samples: %d | Epochs: %d | Batch Size: %d | Learning Rate: %.4f%n",
            totalSamples, epochs, batchSize, learningRate
        );

        double lastAvgLoss = 0.0;
        double lastAccuracy = 0.0;

        for (int epoch = 1; epoch <= epochs; epoch++) {
            if (callback != null) {
                callback.onEpochStart(epoch, epochs);
            }

            // Shuffle dataset every epoch for stochastic generalization
            Collections.shuffle(dataset);

            double epochLoss = 0.0;
            int epochCorrect = 0;

            for (int b = 0; b < totalBatches; b++) {
                int startIdx = b * batchSize;
                int endIdx = Math.min(startIdx + batchSize, totalSamples);

                double batchLoss = 0.0;

                // Accumulate gradients across the mini-batch
                for (int i = startIdx; i < endIdx; i++) {
                    DataSample sample = dataset.get(i);

                    // 1. Forward Pass
                    Tensor prediction = network.forward(sample.getImage());

                    // 2. Calculate Loss
                    double loss = lossFunction.calculateLoss(prediction, sample.getTarget());
                    batchLoss += loss;
                    epochLoss += loss;

                    if (prediction.argMax() == sample.getLabel()) {
                        epochCorrect++;
                    }

                    // 3. Compute Loss Gradient
                    Tensor lossGradient = lossFunction.calculateGradient(prediction, sample.getTarget());

                    // 4. Backward Pass (accumulate gradients in layers)
                    network.backward(lossGradient);
                }

                // 5. Parameter Update (Gradient Descent step)
                network.updateWeights(learningRate);

                if (callback != null) {
                    callback.onBatchEnd(b + 1, totalBatches, batchLoss / (endIdx - startIdx));
                }
            }

            lastAvgLoss = epochLoss / totalSamples;
            lastAccuracy = (double) epochCorrect / totalSamples * 100.0;

            System.out.printf(
                "Epoch [%2d/%2d] -> Loss: %.4f | Accuracy: %6.2f%%%n",
                epoch, epochs, lastAvgLoss, lastAccuracy
            );

            if (callback != null) {
                callback.onEpochEnd(epoch, epochs, lastAvgLoss, lastAccuracy);
            }
        }

        if (callback != null) {
            callback.onTrainingComplete(lastAvgLoss, lastAccuracy);
        }
        System.out.println("[Training Finished Successfully]");
    }

    /**
     * Evaluates the network accuracy on a test dataset without updating weights.
     */
    public EvaluationResult evaluate(List<DataSample> testData) {
        if (testData == null || testData.isEmpty()) {
            throw new IllegalArgumentException("Test dataset cannot be empty.");
        }

        double totalLoss = 0.0;
        int correct = 0;
        int[][] confusionMatrix = new int[10][10];

        for (DataSample sample : testData) {
            Tensor prediction = network.forward(sample.getImage());
            totalLoss += lossFunction.calculateLoss(prediction, sample.getTarget());

            int predictedLabel = prediction.argMax();
            int actualLabel = sample.getLabel();

            if (predictedLabel == actualLabel) {
                correct++;
            }
            if (actualLabel >= 0 && actualLabel < 10 && predictedLabel >= 0 && predictedLabel < 10) {
                confusionMatrix[actualLabel][predictedLabel]++;
            }
        }

        double avgLoss = totalLoss / testData.size();
        double accuracy = (double) correct / testData.size() * 100.0;

        return new EvaluationResult(avgLoss, accuracy, correct, testData.size(), confusionMatrix);
    }

    public static class EvaluationResult {
        public final double averageLoss;
        public final double accuracy;
        public final int correctCount;
        public final int totalCount;
        public final int[][] confusionMatrix;

        public EvaluationResult(double loss, double acc, int correct, int total, int[][] matrix) {
            this.averageLoss = loss;
            this.accuracy = acc;
            this.correctCount = correct;
            this.totalCount = total;
            this.confusionMatrix = matrix;
        }

        public void printReport() {
            System.out.printf(
                "%n--- Evaluation Results ---%nAverage Loss: %.4f%nAccuracy: %d/%d (%.2f%%)%n",
                averageLoss, correctCount, totalCount, accuracy
            );
        }
    }
}
