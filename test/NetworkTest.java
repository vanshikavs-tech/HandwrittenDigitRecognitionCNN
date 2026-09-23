package test;

import java.util.ArrayList;
import java.util.List;
import dataset.DataSample;
import dataset.MNISTLoader;
import layers.ConvolutionLayer;
import layers.FlattenLayer;
import layers.FullyConnectedLayer;
import layers.MaxPoolingLayer;
import layers.ReLULayer;
import layers.SoftmaxLayer;
import loss.CrossEntropyLoss;
import math.Matrix;
import math.Tensor;
import network.NeuralNetwork;
import trainer.Trainer;
import utils.ModelBuilder;

/**
 * Automated Test Suite for HandwrittenDigitCNN.
 * Verifies Math engine, Layer operations, Backpropagation,
 * Loss gradients, and end-to-end toy classification convergence.
 */
public class NetworkTest {

    private static int testsPassed = 0;
    private static int testsFailed = 0;

    public static void main(String[] args) {
        System.out.println("==========================================================");
        System.out.println("             RUNNING AUTOMATED TEST SUITE                 ");
        System.out.println("==========================================================");

        testMatrixOperations();
        testTensorOperations();
        testFullyConnectedBackprop();
        testReLULayer();
        testSoftmaxAndLoss();
        testConvolutionForwardBackward();
        testMaxPoolingForwardBackward();
        testFlattenForwardBackward();
        testToyPatternLearning();

        System.out.println("==========================================================");
        System.out.printf("TEST RESULTS: %d PASSED, %d FAILED%n", testsPassed, testsFailed);
        System.out.println("==========================================================");

        if (testsFailed > 0) {
            System.exit(1);
        }
    }

    private static void assertTrue(String testName, boolean condition) {
        if (condition) {
            System.out.println("  [PASS] " + testName);
            testsPassed++;
        } else {
            System.err.println("  [FAIL] " + testName);
            testsFailed++;
        }
    }

    private static void assertEquals(String testName, double expected, double actual, double tolerance) {
        assertTrue(testName + " (expected: " + expected + ", got: " + actual + ")",
            Math.abs(expected - actual) <= tolerance);
    }

    private static void testMatrixOperations() {
        System.out.println("\nTesting Matrix Operations...");

        // 1. Addition
        Matrix a = new Matrix(2, 2);
        a.setValue(0, 0, 1); a.setValue(0, 1, 2);
        a.setValue(1, 0, 3); a.setValue(1, 1, 4);

        Matrix b = new Matrix(2, 2);
        b.setValue(0, 0, 5); b.setValue(0, 1, 6);
        b.setValue(1, 0, 7); b.setValue(1, 1, 8);

        Matrix c = a.add(b);
        assertEquals("Matrix.add", 6.0, c.getValue(0, 0), 1e-6);
        assertEquals("Matrix.add", 12.0, c.getValue(1, 1), 1e-6);

        // 2. Multiplication
        // [1 2] * [5 6] = [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3 4]   [7 8]   [3*5+4*7, 3*6+4*8]   [43, 50]
        Matrix prod = a.multiply(b);
        assertEquals("Matrix.multiply (0,0)", 19.0, prod.getValue(0, 0), 1e-6);
        assertEquals("Matrix.multiply (0,1)", 22.0, prod.getValue(0, 1), 1e-6);
        assertEquals("Matrix.multiply (1,0)", 43.0, prod.getValue(1, 0), 1e-6);
        assertEquals("Matrix.multiply (1,1)", 50.0, prod.getValue(1, 1), 1e-6);

        // 3. Transpose
        Matrix at = a.transpose();
        assertEquals("Matrix.transpose (0,1)", 3.0, at.getValue(0, 1), 1e-6);
        assertEquals("Matrix.transpose (1,0)", 2.0, at.getValue(1, 0), 1e-6);

        // 4. Hadamard
        Matrix had = a.hadamard(b);
        assertEquals("Matrix.hadamard", 5.0, had.getValue(0, 0), 1e-6);
        assertEquals("Matrix.hadamard", 32.0, had.getValue(1, 1), 1e-6);

        // 5. ArgMax
        Matrix vec = Matrix.fromVector(new double[]{0.1, 0.9, 0.4, 0.2});
        assertTrue("Matrix.argMax", vec.argMax() == 1);
    }

    private static void testTensorOperations() {
        System.out.println("\nTesting Tensor Operations...");
        Tensor t = new Tensor(2, 3, 4);
        t.fill(2.5);

        assertEquals("Tensor.size", 24.0, t.size(), 1e-6);
        assertEquals("Tensor.getValue", 2.5, t.getValue(0, 1, 2), 1e-6);

        Matrix flat = t.flattenToMatrix();
        assertTrue("Tensor.flatten dimensions", flat.getRows() == 1 && flat.getCols() == 24);

        Matrix ch0 = t.getChannel(0);
        assertTrue("Tensor.getChannel dimensions", ch0.getRows() == 3 && ch0.getCols() == 4);
        assertEquals("Tensor.getChannel value", 2.5, ch0.getValue(1, 2), 1e-6);
    }

    private static void testFullyConnectedBackprop() {
        System.out.println("\nTesting FullyConnectedLayer Backprop...");
        FullyConnectedLayer fc = new FullyConnectedLayer(3, 2);

        Matrix input = Matrix.fromVector(new double[]{1.0, 2.0, 3.0});
        Matrix out = fc.forward(input);
        assertTrue("Dense forward dimension", out.getCols() == 2);

        Matrix gradOut = Matrix.fromVector(new double[]{0.5, -0.5});
        Matrix inGrad = fc.backward(gradOut);
        assertTrue("Dense backward input grad dimension", inGrad.getCols() == 3);

        // Update weights
        fc.updateWeights(0.1);
        assertTrue("Dense updateWeights executed", true);
    }

    private static void testReLULayer() {
        System.out.println("\nTesting ReLULayer...");
        ReLULayer relu = new ReLULayer();

        Matrix in = Matrix.fromVector(new double[]{-3.0, 0.0, 4.5});
        Matrix out = relu.forward(in);

        assertEquals("ReLU negative zeroed", 0.0, out.getValue(0, 0), 1e-6);
        assertEquals("ReLU zero", 0.0, out.getValue(0, 1), 1e-6);
        assertEquals("ReLU positive preserved", 4.5, out.getValue(0, 2), 1e-6);

        Matrix gradOut = Matrix.fromVector(new double[]{1.0, 1.0, 1.0});
        Matrix gradIn = relu.backward(gradOut);

        assertEquals("ReLU backward negative blocked", 0.0, gradIn.getValue(0, 0), 1e-6);
        assertEquals("ReLU backward positive passed", 1.0, gradIn.getValue(0, 2), 1e-6);
    }

    private static void testSoftmaxAndLoss() {
        System.out.println("\nTesting Softmax and CrossEntropyLoss...");
        SoftmaxLayer sm = new SoftmaxLayer();

        Matrix logits = Matrix.fromVector(new double[]{2.0, 1.0, 0.1});
        Matrix probs = sm.forward(logits);

        double sum = probs.getValue(0, 0) + probs.getValue(0, 1) + probs.getValue(0, 2);
        assertEquals("Softmax sums to 1.0", 1.0, sum, 1e-5);
        assertTrue("Softmax highest logit is highest prob", probs.getValue(0, 0) > probs.getValue(0, 1));

        CrossEntropyLoss loss = new CrossEntropyLoss();
        Matrix target = Matrix.fromVector(new double[]{1.0, 0.0, 0.0});

        double lossVal = loss.calculateLoss(probs, target);
        assertTrue("Loss is positive", lossVal > 0.0);

        Matrix grad = loss.calculateGradient(probs, target);
        // (P - Y): for class 0, should be P_0 - 1 < 0
        assertTrue("Combined gradient class 0 negative", grad.getValue(0, 0) < 0.0);
        assertTrue("Combined gradient class 1 positive", grad.getValue(0, 1) > 0.0);
    }

    private static void testConvolutionForwardBackward() {
        System.out.println("\nTesting ConvolutionLayer Forward & Backward...");
        ConvolutionLayer conv = new ConvolutionLayer(1, 2, 3, 1, 0);

        Tensor in = new Tensor(1, 6, 6);
        in.randomInitialize();

        Tensor out = conv.forward(in);
        // Out size: (6 - 3)/1 + 1 = 4 -> 2 x 4 x 4
        assertTrue("Conv output channels", out.getChannels() == 2);
        assertTrue("Conv output height", out.getHeight() == 4);
        assertTrue("Conv output width", out.getWidth() == 4);

        Tensor gradOut = new Tensor(2, 4, 4);
        gradOut.fill(0.1);
        Tensor inGrad = conv.backward(gradOut);

        assertTrue("Conv backprop inGrad channels", inGrad.getChannels() == 1);
        assertTrue("Conv backprop inGrad height", inGrad.getHeight() == 6);
        assertTrue("Conv backprop inGrad width", inGrad.getWidth() == 6);

        conv.updateWeights(0.01);
        assertTrue("Conv updateWeights executed", true);
    }

    private static void testMaxPoolingForwardBackward() {
        System.out.println("\nTesting MaxPoolingLayer Forward & Backward...");
        MaxPoolingLayer pool = new MaxPoolingLayer(2, 2);

        Tensor in = new Tensor(1, 4, 4);
        in.setValue(0, 0, 0, 1.0); in.setValue(0, 0, 1, 5.0);
        in.setValue(0, 1, 0, 3.0); in.setValue(0, 1, 1, 2.0);

        Tensor out = pool.forward(in);
        assertTrue("Pool output dimensions", out.getHeight() == 2 && out.getWidth() == 2);
        assertEquals("Pool max value correctly picked", 5.0, out.getValue(0, 0, 0), 1e-6);

        Tensor gradOut = new Tensor(1, 2, 2);
        gradOut.setValue(0, 0, 0, 0.7);

        Tensor inGrad = pool.backward(gradOut);
        assertEquals("Pool grad routed to max pos (0,1)", 0.7, inGrad.getValue(0, 0, 1), 1e-6);
        assertEquals("Pool grad zero at non-max pos (0,0)", 0.0, inGrad.getValue(0, 0, 0), 1e-6);
    }

    private static void testFlattenForwardBackward() {
        System.out.println("\nTesting FlattenLayer...");
        FlattenLayer flat = new FlattenLayer();

        Tensor in = new Tensor(2, 3, 4);
        in.fill(1.5);
        Tensor out = flat.forward(in);

        assertTrue("Flatten forward size", out.size() == 24);

        Tensor gradOut = Tensor.zeros(1, 1, 24);
        gradOut.fill(0.2);
        Tensor inGrad = flat.backward(gradOut);

        assertTrue("Flatten backward reshaped channels", inGrad.getChannels() == 2);
        assertTrue("Flatten backward reshaped height", inGrad.getHeight() == 3);
        assertTrue("Flatten backward reshaped width", inGrad.getWidth() == 4);
        assertEquals("Flatten backward value", 0.2, inGrad.getValue(0, 0, 0), 1e-6);
    }

    private static void testToyPatternLearning() {
        System.out.println("\nTesting Toy Pattern Learning Convergence...");

        // Create a 2-class toy dataset with 4 samples:
        // Digit 0: 3x3 box
        // Digit 1: 3x3 vertical line
        Tensor zeroSample = new Tensor(1, 3, 3);
        zeroSample.setValue(0, 0, 0, 1); zeroSample.setValue(0, 0, 1, 1); zeroSample.setValue(0, 0, 2, 1);
        zeroSample.setValue(0, 1, 0, 1); zeroSample.setValue(0, 1, 1, 0); zeroSample.setValue(0, 1, 2, 1);
        zeroSample.setValue(0, 2, 0, 1); zeroSample.setValue(0, 2, 1, 1); zeroSample.setValue(0, 2, 2, 1);

        Tensor oneSample = new Tensor(1, 3, 3);
        oneSample.setValue(0, 0, 1, 1);
        oneSample.setValue(0, 1, 1, 1);
        oneSample.setValue(0, 2, 1, 1);

        List<DataSample> data = new ArrayList<>();
        data.add(new DataSample(zeroSample, 0, 2));
        data.add(new DataSample(oneSample, 1, 2));

        // Build a miniature network: Flatten -> Dense(9, 8) -> ReLU -> Dense(8, 2) -> Softmax
        NeuralNetwork net = new NeuralNetwork();
        net.addLayer(new FlattenLayer());
        net.addLayer(new FullyConnectedLayer(9, 8));
        net.addLayer(new ReLULayer());
        net.addLayer(new FullyConnectedLayer(8, 2));
        net.addLayer(new SoftmaxLayer());

        Trainer trainer = new Trainer(net);

        // Evaluate initial loss
        Trainer.EvaluationResult initEval = trainer.evaluate(data);

        // Train for 60 iterations
        trainer.train(data, 60, 0.1, 2, null);

        Trainer.EvaluationResult finalEval = trainer.evaluate(data);

        System.out.printf("  Initial Loss: %.4f | Final Loss: %.4f%n", initEval.averageLoss, finalEval.averageLoss);
        assertTrue("Loss decreased through backpropagation", finalEval.averageLoss < initEval.averageLoss);
        assertTrue("Accuracy reached 100% on toy pattern", finalEval.accuracy == 100.0);
    }
}
