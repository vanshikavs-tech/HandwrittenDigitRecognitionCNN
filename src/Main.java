import java.awt.GraphicsEnvironment;
import java.io.File;
import java.util.List;
import javax.swing.SwingUtilities;
import dataset.DataSample;
import dataset.MNISTLoader;
import gui.DigitVisualizer;
import gui.MainWindow;
import model.ModelSerializer;
import network.NeuralNetwork;
import test.NetworkTest;
import trainer.Trainer;
import utils.ModelBuilder;

/**
 * Main application entry point for HandwrittenDigitCNN.
 *
 * Usage:
 *   java Main                -> Launches interactive Swing GUI (or demo if headless)
 *   java Main --gui          -> Launches interactive Swing GUI
 *   java Main --test         -> Runs automated unit test suite
 *   java Main --demo         -> Trains quick demo CNN on synthetic digits and prints predictions
 *   java Main --train [num]  -> Trains CNN on MNIST dataset (default: 2000 samples)
 */
public class Main {

    public static void main(String[] args) {
        String mode = (args.length > 0) ? args[0].toLowerCase() : "--default";

        switch (mode) {
            case "--test":
                NetworkTest.main(args);
                break;

            case "--demo":
                runQuickDemo();
                break;

            case "--train":
                int samples = (args.length > 1) ? Integer.parseInt(args[1]) : 2000;
                int epochs = (args.length > 2) ? Integer.parseInt(args[2]) : 5;
                trainOnMnist(samples, epochs);
                break;

            case "--gui":
                launchGui();
                break;

            case "--default":
            default:
                if (!GraphicsEnvironment.isHeadless()) {
                    launchGui();
                } else {
                    System.out.println("Headless environment detected. Running CLI Demo...");
                    runQuickDemo();
                }
                break;
        }
    }

    private static void launchGui() {
        System.out.println("Launching Handwritten Digit Recognition GUI...");
        SwingUtilities.invokeLater(() -> {
            NeuralNetwork network = null;
            if (ModelSerializer.defaultModelExists()) {
                try {
                    network = ModelSerializer.loadModel(ModelSerializer.DEFAULT_MODEL_PATH);
                    System.out.println("Loaded pretrained model from " + ModelSerializer.DEFAULT_MODEL_PATH);
                } catch (Exception e) {
                    System.out.println("Could not load default model, initializing new CNN: " + e.getMessage());
                }
            }

            if (network == null) {
                network = ModelBuilder.buildCNN();
            }

            MainWindow window = new MainWindow(network);
            window.setVisible(true);
        });
    }

    private static void runQuickDemo() {
        System.out.println("==========================================================");
        System.out.println("        HANDWRITTEN DIGIT CNN - QUICK DEMO MODE          ");
        System.out.println("==========================================================");

        // 1. Build CNN Architecture
        NeuralNetwork cnn = ModelBuilder.buildFastCNN();
        cnn.printSummary();

        // 2. Generate synthetic digit dataset for offline demo
        System.out.println("\nGenerating synthetic training digits (0-9)...");
        List<DataSample> trainData = MNISTLoader.createSyntheticDataset(30); // 300 samples
        List<DataSample> testData = MNISTLoader.createSyntheticDataset(10);  // 100 samples

        // 3. Train network
        Trainer trainer = new Trainer(cnn);
        trainer.train(trainData, 5, 0.02, 16, null);

        // 4. Evaluate on test data
        System.out.println("\nEvaluating on unseen test set:");
        Trainer.EvaluationResult eval = trainer.evaluate(testData);
        eval.printReport();

        // 5. Visualize sample predictions
        System.out.println("\n--- Testing Sample Digits with ASCII Visualization ---");
        for (int digit = 0; digit < Math.min(3, testData.size()); digit++) {
            DataSample sample = testData.get(digit * 10);
            System.out.printf("%nActual Digit: %d%n", sample.getLabel());
            DigitVisualizer.printAscii(sample.getImage());

            int predicted = cnn.predictDigit(sample.getImage());
            double confidence = cnn.predictConfidence(sample.getImage());
            System.out.printf("CNN Predicted Digit: %d (Confidence: %.1f%%)%n", predicted, confidence * 100.0);
        }

        // Save demo model so GUI has immediate access
        try {
            ModelSerializer.saveModel(cnn, ModelSerializer.DEFAULT_MODEL_PATH);
        } catch (Exception e) {
            System.err.println("Note: could not save demo model: " + e.getMessage());
        }

        System.out.println("\nDemo complete! Run 'java Main --gui' to test in the interactive canvas.");
    }

    private static void trainOnMnist(int samples, int epochs) {
        System.out.println("==========================================================");
        System.out.printf("  TRAINING CNN ON MNIST (%d samples, %d epochs)%n", samples, epochs);
        System.out.println("==========================================================");

        try {
            List<DataSample> trainData = MNISTLoader.loadTrainingData(samples);
            List<DataSample> testData = MNISTLoader.loadTestData(Math.min(1000, samples / 2));

            NeuralNetwork cnn = ModelBuilder.buildCNN();
            cnn.printSummary();

            Trainer trainer = new Trainer(cnn);
            trainer.train(trainData, epochs, 0.01, 32, null);

            Trainer.EvaluationResult eval = trainer.evaluate(testData);
            eval.printReport();

            ModelSerializer.saveModel(cnn, ModelSerializer.DEFAULT_MODEL_PATH);
            System.out.println("Model saved! You can now launch the GUI with: java Main --gui");
        } catch (Exception e) {
            System.err.println("Training error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}